from langgraph.checkpoint.postgres import PostgresSaver
from src.settings import settings
from src.graph.graph import graph

# Global compiled app instance
_compiled_app = None
_checkpointer_context = None


def build_app():
    global _compiled_app, _checkpointer_context

    if _compiled_app is None:
        try:
            print(f"🔄 Connecting to database: {settings.AWS_DB_URL[:50]}...")

            # Create the context manager but keep it alive
            _checkpointer_context = PostgresSaver.from_conn_string(settings.AWS_DB_URL)
            print("🔄 Context manager created, entering context...")

            # Enter the context to get the actual checkpointer
            checkpointer = _checkpointer_context.__enter__()
            print("🔄 Context entered, setting up tables...")

            try:
                checkpointer.setup()  # Create tables if they don't exist
                print("✅ Database tables created/verified")
            except Exception as setup_error:
                print(f"⚠️  Setup warning (probably tables exist): {setup_error}")

            print("🔄 Compiling graph...")
            _compiled_app = graph.compile(checkpointer=checkpointer)
            print("✅ Using PostgresSaver for persistent sessions")

        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("🔄 Check your database connection and security groups")
            raise e

    return _compiled_app


def cleanup():
    """Call this on server shutdown"""
    global _checkpointer_context
    if _checkpointer_context:
        try:
            _checkpointer_context.__exit__(None, None, None)
        except:
            pass


# For testing the module directly
if __name__ == "__main__":
    print("Testing database connection...")
    try:
        app = build_app()
        print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

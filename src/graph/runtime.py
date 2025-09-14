from src.settings import settings
from src.graph.graph import graph
from src.db.checkpointer import create_checkpointer_context
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.db.url import get_libpq_url

# Global compiled app instance
_compiled_app = None
_checkpointer_context = None
_async_checkpointer_context = None
_async_checkpointer = None


def build_app():
    global _compiled_app, _checkpointer_context

    if _compiled_app is None:
        # Try PostgresSaver first
        try:
            print(f"🔄 Connecting to database: {settings.AWS_DB_URL[:50]}...")

            # Create the context manager but keep it alive
            _checkpointer_context = create_checkpointer_context()
            print("🔄 Context manager created, entering context...")

            # Enter the context to get the actual checkpointer
            checkpointer = _checkpointer_context.__enter__()
            print("🔄 Context entered, setting up tables...")

            try:
                checkpointer.setup()  # Create tables if they don't exist
                print("✅ Database tables created/verified")
            except Exception as setup_error:
                print(f"⚠️  Setup warning (probably tables exist): {setup_error}")

            # Detect async support; astream requires async checkpointer methods
            supports_async = all(
                hasattr(checkpointer, attr)
                for attr in ("aget_tuple", "aput", "adelete")
            )
            if not supports_async:
                print(
                    "⚠️  PostgresSaver lacks async checkpointing; falling back to MemorySaver for streaming"
                )
                # Exit context; we'll not use the Postgres checkpointer
                try:
                    _checkpointer_context.__exit__(None, None, None)
                except Exception:
                    pass
                _checkpointer_context = None
                memory_cp = MemorySaver()
                _compiled_app = graph.compile(checkpointer=memory_cp)
                print("✅ Using MemorySaver (in-memory) for sessions")
            else:
                print("🔄 Compiling graph with PostgresSaver...")
                _compiled_app = graph.compile(checkpointer=checkpointer)
                print("✅ Using PostgresSaver for persistent sessions")

        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print("🔄 Falling back to MemorySaver (no persistence)")
            memory_cp = MemorySaver()
            _compiled_app = graph.compile(checkpointer=memory_cp)
            print("✅ Using MemorySaver for sessions")

    return _compiled_app


async def build_app_async():
    global _compiled_app, _async_checkpointer_context, _async_checkpointer

    if _compiled_app is None:
        try:
            print(f"🔄 Connecting to database (async): {settings.AWS_DB_URL[:50]}...")
            # Keep async context alive for app lifetime
            _async_checkpointer_context = AsyncPostgresSaver.from_conn_string(
                get_libpq_url()
            )
            _async_checkpointer = await _async_checkpointer_context.__aenter__()

            # Setup tables (async if available)
            try:
                if hasattr(_async_checkpointer, "asetup"):
                    await _async_checkpointer.asetup()
                else:
                    _async_checkpointer.setup()
                print("✅ Database tables created/verified (async)")
            except Exception as setup_error:
                print(f"⚠️  Setup warning: {setup_error}")

            print("🔄 Compiling graph with AsyncPostgresSaver...")
            _compiled_app = graph.compile(checkpointer=_async_checkpointer)
            print("✅ Using AsyncPostgresSaver for persistent sessions")
        except Exception as e:
            print(f"❌ Async database connection failed: {e}")
            print("🔄 Falling back to sync builder or MemorySaver")
            # Fallback to sync builder (which may choose MemorySaver)
            return build_app()

    return _compiled_app


def cleanup():
    """Call this on server shutdown"""
    global _checkpointer_context
    if _checkpointer_context:
        try:
            _checkpointer_context.__exit__(None, None, None)
        except:
            pass


async def acleanup():
    global _async_checkpointer_context
    if _async_checkpointer_context:
        try:
            await _async_checkpointer_context.__aexit__(None, None, None)
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

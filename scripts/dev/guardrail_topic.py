# Import Guard and Validator
from guardrails.hub import RestrictToTopic
from guardrails import Guard

# Setup Guard
guard = Guard().use(
    RestrictToTopic(
        valid_topics=[
            "pc parts support",
            "orders",
            "hardware compatibility",
            "warranty",
            "returns",
        ],
        disable_classifier=False,
        disable_llm=True,
        on_fail="fix",
    )
)

print(
    guard.validate("""
I have a question about my order.
""")
)  # Validator passes

print(
    guard.validate("""
What is the history of the colonization of the Philippines?
""")
)  # Validator fails

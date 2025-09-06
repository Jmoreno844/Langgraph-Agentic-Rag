# Import Guard and Validator
from guardrails.hub import DetectPII
from guardrails import Guard


# Setup Guard
guard = Guard().use(DetectPII, ["EMAIL_ADDRESS", "PHONE_NUMBER"], "exception")

guard.validate("Good morning!")  # Validator passes
try:
    guard.validate("Ok, my mail is , let me know when there are new discounts")
    print("Validated")
except Exception as e:
    print(e)

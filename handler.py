import logging
from model import Model

# configure logger and initialize model outside of the handle() function

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

model = Model()

def handle(event, context):
    if event.get("source") == "KEEP_LAMBDA_WARM":
        LOGGER.info("No ML work to do. Just staying warm...")
        return "Keeping Lambda warm"

    return {
        "sentiment": model.predict(text=event["text"])
    }
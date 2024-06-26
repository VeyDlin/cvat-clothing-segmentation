# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init ModelHandler...  0%")
    model = ModelHandler(context)
    context.user_data.model = model
    context.logger.info("Init ModelHandler...100%")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB

    context.logger.info(f"context.user_data.model.handle")
    features = context.user_data.model.handle(image)

    context.logger.info(f"features")
    return context.Response(body=json.dumps({
            'blob': base64.b64encode((features.cpu().numpy() if features.is_cuda else features.numpy())).decode(),
        }),
        headers={},
        content_type='application/json',
        status_code=200
    )

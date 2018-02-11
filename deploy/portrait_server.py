# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import time
import grpc

import portrait_pb2
import portrait_pb2_grpc
from inference import TF_Portrait

import numpy as np
import cv2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class PortraitService(portrait_pb2_grpc.PortraitBackendServicer):
    def __init__(self):
        self.m_portrait = TF_Portrait('./model.json', './head-segmentation-model.h5')

    def Portrait(self, request, context):
        w, h = request.rect.w, request.rect.h
        img = np.fromstring(request.input_image, dtype=np.uint8).reshape(h, w, 3)
        print(img.shape)

        result = self.m_portrait.inference(img)

        response = portrait_pb2.PortraitResponse()
        response.completion_image = result.tobytes()
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    portrait_pb2_grpc.add_PortraitBackendServicer_to_server(PortraitService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
  serve()
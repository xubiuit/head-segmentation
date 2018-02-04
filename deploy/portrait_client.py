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

"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import grpc

import portrait_pb2
import portrait_pb2_grpc

import numpy as np
import cv2
import time

def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = portrait_pb2_grpc.PortraitBackendStub(channel)

  request = portrait_pb2.PortraitRequest()

  img = cv2.imread('00002.png')
  request.input_image = img.tobytes()

  request.rect.h, request.rect.w = img.shape[:2]

  response = stub.Portrait(request)
  response = np.fromstring(response.completion_image, dtype=np.uint8).reshape(img.shape[0], img.shape[1], 4)

  # save response image for verification
  cv2.imwrite('tmp-res.png', response)

if __name__ == '__main__':
  run()
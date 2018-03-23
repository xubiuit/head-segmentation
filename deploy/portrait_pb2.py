# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: portrait.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='portrait.proto',
  package='portrait',
  syntax='proto3',
  serialized_pb=_b('\n\x0eportrait.proto\x12\x08portrait\"\x1c\n\x04Rect\x12\t\n\x01w\x18\x01 \x01(\x05\x12\t\n\x01h\x18\x02 \x01(\x05\"X\n\x0fPortraitRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x13\n\x0binput_image\x18\x02 \x01(\x0c\x12\x1c\n\x04rect\x18\x03 \x01(\x0b\x32\x0e.portrait.Rect\",\n\x10PortraitResponse\x12\x18\n\x10\x63ompletion_image\x18\x01 \x01(\x0c\x32V\n\x0fPortraitBackend\x12\x43\n\x08Portrait\x12\x19.portrait.PortraitRequest\x1a\x1a.portrait.PortraitResponse\"\x00\x62\x06proto3')
)




_RECT = _descriptor.Descriptor(
  name='Rect',
  full_name='portrait.Rect',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='w', full_name='portrait.Rect.w', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='h', full_name='portrait.Rect.h', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=56,
)


_PORTRAITREQUEST = _descriptor.Descriptor(
  name='PortraitRequest',
  full_name='portrait.PortraitRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_id', full_name='portrait.PortraitRequest.request_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='input_image', full_name='portrait.PortraitRequest.input_image', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rect', full_name='portrait.PortraitRequest.rect', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=146,
)


_PORTRAITRESPONSE = _descriptor.Descriptor(
  name='PortraitResponse',
  full_name='portrait.PortraitResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='completion_image', full_name='portrait.PortraitResponse.completion_image', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=148,
  serialized_end=192,
)

_PORTRAITREQUEST.fields_by_name['rect'].message_type = _RECT
DESCRIPTOR.message_types_by_name['Rect'] = _RECT
DESCRIPTOR.message_types_by_name['PortraitRequest'] = _PORTRAITREQUEST
DESCRIPTOR.message_types_by_name['PortraitResponse'] = _PORTRAITRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Rect = _reflection.GeneratedProtocolMessageType('Rect', (_message.Message,), dict(
  DESCRIPTOR = _RECT,
  __module__ = 'portrait_pb2'
  # @@protoc_insertion_point(class_scope:portrait.Rect)
  ))
_sym_db.RegisterMessage(Rect)

PortraitRequest = _reflection.GeneratedProtocolMessageType('PortraitRequest', (_message.Message,), dict(
  DESCRIPTOR = _PORTRAITREQUEST,
  __module__ = 'portrait_pb2'
  # @@protoc_insertion_point(class_scope:portrait.PortraitRequest)
  ))
_sym_db.RegisterMessage(PortraitRequest)

PortraitResponse = _reflection.GeneratedProtocolMessageType('PortraitResponse', (_message.Message,), dict(
  DESCRIPTOR = _PORTRAITRESPONSE,
  __module__ = 'portrait_pb2'
  # @@protoc_insertion_point(class_scope:portrait.PortraitResponse)
  ))
_sym_db.RegisterMessage(PortraitResponse)



_PORTRAITBACKEND = _descriptor.ServiceDescriptor(
  name='PortraitBackend',
  full_name='portrait.PortraitBackend',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=194,
  serialized_end=280,
  methods=[
  _descriptor.MethodDescriptor(
    name='Portrait',
    full_name='portrait.PortraitBackend.Portrait',
    index=0,
    containing_service=None,
    input_type=_PORTRAITREQUEST,
    output_type=_PORTRAITRESPONSE,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_PORTRAITBACKEND)

DESCRIPTOR.services_by_name['PortraitBackend'] = _PORTRAITBACKEND

# @@protoc_insertion_point(module_scope)
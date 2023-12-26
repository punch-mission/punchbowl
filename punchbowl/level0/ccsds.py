# from __future__ import annotations
#
# import logging
# import typing as t
#
# import ccsdspy
# import numpy as np
# import pandas as pd
#
# log = logging.getLogger(__name__)
#
#
# APID_TO_PACKET_NAME= {32: "SCI_XFI"}
#
# # class SliceableInt(int):
# #     """Integer that can be indexed bit-by-bit
# #
# #     Written by Caden Gobat
# #     """
# #
# #     size: int  # measured in bytes
# #
# #     def __new__(cls, value: int, size: int = None) -> "SliceableInt":
# #         if isinstance(value, bytes):
# #             return cls.from_bytes(value)
# #         instance = super().__new__(cls, value)
# #         if size is None:
# #             log.warning(f"{cls.__name__} parameter `size` was not provided. This means the `len()`"
# #                         f" of this object will be the minimum number of bits required to represent"
# #                         f" the quantity {instance:d}, even if that is not a multiple of 8 (i.e., a"
# #                         " whole number of bytes)")
# #         instance.size = size
# #         return instance
# #
# #     @classmethod
# #     def from_bytes(cls, b:bytes, byteorder: str = "big") -> SliceableInt:
# #         return cls(int.from_bytes(b, byteorder), size=len(b))
# #
# #     def __str__(self) -> str:
# #         return int(self).__str__()
# #
# #     def __repr__(self) -> str:
# #         return int(self).__repr__()
# #
# #     def __len__(self) -> int:
# #         if self.size:
# #             return self.size * 8  # if initialized from bytes, we had a set number of them
# #         elif int(self):
# #             return math.ceil(math.log2(int(self)))  # min number of bits required (i.e. no leading 0s)
# #         else:  # value is 0
# #             return 1  # takes 1 bit to represent
# #
# #     def __getitem__(self, idx) -> int:
# #         """Provides access to individual bits. Uses CCSDS ordering conventions (MSb is bit 0).
# #
# #         From 23103-uSat2Gnd_ICD-01:
# #         "Per CCSDS conventions, the first bit in a field to be transmitted (i.e., the most left
# #          justified when drawing a figure) is defined to be “Bit 0”; the following bit is defined
# #          to be “Bit 1” and so on up to “Bit N-1”. When the field is used to express a binary value
# #          (such as a counter), the most significant bit (MSb) shall be the first transmitted bit of
# #          the field, i.e., “Bit 0”."
# #         """
# #         if isinstance(idx, t.SupportsIndex):  # single-bit extraction (int or int-like)
# #             if idx < 0:
# #                 raise IndexError("Negative bit indices are not supported.")
# #             if idx > len(self):
# #                 raise IndexError(f"Index {idx} out of range for {type(self).__name__} "
# #                                  f"of length {len(self)} bits.")
# #             bit_value = (int(self) >> (len(self) - 1 - idx)) & 1
# #             return self.__class__(bit_value, size=1/8)  # single bit is 1/8 of a byte
# #         elif isinstance(idx, slice):
# #             if idx.start < 0 or idx.stop < 0:
# #                 raise IndexError("Negative bit indices are not supported.")
# #
# #             start = 0 if idx.start is None else idx.start
# #             stop = len(self) if idx.stop is None else idx.stop
# #
# #             if idx.step not in (None, 1) or start > stop:
# #                 raise IndexError("Bit slicing with step size !=1 (incl. reverse slices) are not supported")
# #
# #             slice_len = stop - start
# #             extra_bits_to_right = len(self) - stop
# #             sliced_value = (int(self) >> extra_bits_to_right) & (2 ** slice_len - 1)
# #
# #             return self.__class__(sliced_value, size=slice_len / 8)
# #         else:
# #             raise TypeError(f"Bit indices must be integers or slices, not {type(idx)}")
# #
# #
# # class ByteField(bytes):
# #     """Represents a data field of bytes.
# #
# #     Main feature is better rendering that circumvents Python's attempt to show bytes
# #     that map to ASCII chars as those characters. Also provides a `.bits` attribute that
# #     enables access to a `SliceableInt` object representing the numeric value, that can
# #     also be indexed bit-by-bit
# #
# #     Written by Caden Gobat
# #     """
# #
# #     bits: SliceableInt
# #     byteorder: str
# #
# #     def __new__(cls, raw, byteorder: str = "big") -> ByteField:
# #         bytefield = super().__new__(cls, bytes(raw))
# #         bytefield.byteorder = byteorder
# #         return bytefield
# #
# #     def __str__(self) -> str:
# #         if len(self):
# #             hex_str = self.hex(sep="|", bytes_per_sep=1).upper()
# #             formatted = hex_str.replace("|", r"\x")  # necessary to avoid capitalization of the '\x'
# #             return r"b'\x" + formatted + "'"
# #         else:
# #             return "b''"
# #
# #     def __repr__(self) -> str:
# #         return "ByteField(" + str(self) + ")"
# #
# #     @property
# #     def bits(self) -> SliceableInt:
# #         if not hasattr(self, "_bits"):
# #             self._bits = SliceableInt.from_bytes(bytes(self), byteorder=self.byteorder)
# #         return self._bits
# #
# #     # A `@bits.setter` method is purposefully omitted, since this object should be immutable
# #
# #     # `def __int__(self):` is purposefully ommitted because it would unnecessarily duplicate
# #     # functionality that `int.from_bytes(bytefield)`` and bytefield.bits both already provide
# #
# #     def __getitem__(self, idx) -> "ByteField|int":
# #         if isinstance(idx, int):
# #             return bytes(self)[idx]
# #         else:
# #             return self.__class__(bytes(self)[idx])
# #
# #     def __lshift__(self, nbits: int):
# #         return self.bits << nbits
# #
# #     def __rshift__(self, nbits: int):
# #         return self.bits >> nbits
# #
# #
# # def get_compression_settings(com_set_val: t.Union[bytes, int]) -> t.Dict[str, int]:
# #     if isinstance(com_set_val, bytes):
# #         compress_config = ByteField(com_set_val)
# #     elif isinstance(com_set_val, (int, np.integer)):
# #         compress_config = ByteField(int(com_set_val).to_bytes(2, "big"))
# #     else:
# #         raise TypeError
# #     settings_dict = {"SCALE":    compress_config.bits[0:8],
# #                      "RSVD":     compress_config.bits[8],
# #                      "PMB_INIT": compress_config.bits[9],
# #                      "CMP_BYP":  compress_config.bits[10],
# #                      "BSEL":     compress_config.bits[11:13],
# #                      "SQRT":     compress_config.bits[13],
# #                      "JPEG":     compress_config.bits[14],
# #                      "TEST":     compress_config.bits[15]}
# #     return settings_dict
# #
# #
# # def get_acquisition_settings(acq_set_val: t.Union[bytes, int]) -> t.Dict[str, int]:
# #     if isinstance(acq_set_val, bytes):
# #         acquire_config = ByteField(acq_set_val)
# #     elif isinstance(acq_set_val, (int, np.integer)):
# #         acquire_config = ByteField(int(acq_set_val).to_bytes(4, "big"))
# #     else:
# #         raise TypeError
# #     settings_dict = {"DELAY":    acquire_config.bits[0:8],
# #                      "IMG_NUM":  acquire_config.bits[8:11],
# #                      "EXPOSURE": acquire_config.bits[11:24],
# #                      "TABLE1":   acquire_config.bits[24:28],
# #                      "TABLE2":   acquire_config.bits[28:32]}
# #     return settings_dict
# #
# #
# # def unpack_nbit_values(packed: bytes, byteorder: str, N=19) -> np.ndarray:
# #     if N in (8, 16, 32, 64):
# #         trailing = len(packed)%(N//8)
# #         if trailing:
# #             log.debug(f"Truncating {trailing} extra bytes")
# #             packed = packed[:-trailing]
# #         return np.frombuffer(packed, dtype=np.dtype(f"u{N//8}").newbyteorder(byteorder))
# #     nbits = len(packed)*8
# #     bytes_as_ints = np.frombuffer(packed, "u1")
# #     results = []
# #     for bit in range(0, nbits, N):
# #         encompassing_bytes = bytes_as_ints[bit//8:-((bit+N)//-8)]
# #         # "ceil" equivalent of a//b is -(-a//b), because of
# #         # http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html
# #         if len(encompassing_bytes)*8 < N:
# #             log.debug(f"Terminating at bit {bit} because there are only {len(encompassing_bytes)*8}"
# #                       f" bits left, which is not enough to make a {N}-bit value.")
# #             break
# #         bit_within_byte = bit % 8
# #         bytes_value = 0
# #         if byteorder in ("little", "<"):
# #             bytes_value = int.from_bytes(encompassing_bytes, "little")
# #             bits_value = (bytes_value >> bit_within_byte) & (2**N - 1)
# #         elif byteorder in ("big", ">"):
# #             extra_bits_to_right = len(encompassing_bytes)*8 - (bit_within_byte+N)
# #             bytes_value = int.from_bytes(encompassing_bytes, "big")
# #             bits_value = (bytes_value >> extra_bits_to_right) & (2**N - 1)
# #         else:
# #             raise ValueError("`byteorder` must be either 'little' or 'big'")
# #         results.append(bits_value)
# #     return np.asanyarray(results)
# #
# #
# # def get_img_array(img_packets: "PacketArchive[SciPacket]", restore=True, width=2048):
# #     com_set_values = list(img_packets["img_com_set"].unique())
# #     img_com_set = int(com_set_values.pop())
# #     if len(com_set_values):  # len should be 0 after pop() if only 1 value was originally present
# #         raise ValueError("Image packets do not have uniform compression settings. "
# #                          "Cannot determine decoding procedure.")
# #     compression_settings = get_compression_settings(img_com_set) # dictionary of unpacked settings
# #     log.debug(f"Processing {len(img_packets)} packets with register "
# #               f"x$9A0={img_com_set} {dict(compression_settings)}")
# #     if compression_settings["JPEG"]:  # JPEG bit enabled (upper two pathways)
# #         if compression_settings["CMP_BYP"]:  # skipped actual JPEG-ification
# #             pixel_values = unpack_nbit_values(img_packets["data"].sum(), byteorder=">", N=16)
# #             # either 12-bit values, but placed into 16b words
# where the 4 MSb are 0000; or 16-bit truncated pixel values
# #         else:  # data is in JPEG-LS format
# #             jpeg_packets = JpegImage.frompackets(img_packets)
# #             pixel_values: np.ndarray = pylibjpeg.decode(jpeg_packets.data)
# #         if compression_settings["SQRT"] and restore:  # SQRT bit enabled
# #             pixel_values = np.square(pixel_values.astype(int))
# #         if compression_settings["SCALE"] and restore:  # data has a non-zero SCALE factor
# #             pixel_values = np.true_divide(pixel_values,
# #                                           compression_settings["SCALE"],
# #                                           casting="safe")
# #     else:
# #         pixel_values = unpack_nbit_values(img_packets["data"].sum(), byteorder="<", N=19)
# #     nvals = pixel_values.size
# #     if nvals % width == 0:
# #         return pixel_values.reshape((-1, width))
# #     else:
# #         return np.ravel(pixel_values)[:width*(nvals//width)].reshape((-1, width))
# #
# #
# # class ImagePackets:
# #     # TODO: reimplement based on CCSDSPY output
# #     def __init__(self, *args, **kwargs) -> None:
# #         super().__init__(*args, **kwargs)
# #
# #     @classmethod
# #     def frompackets(cls, packets):
# #         complete_sci = []
# #
# #         for pkt in packets:
# #             if pkt["ap_id"] == 32:
# #                 complete_sci.append(SciPacket(pkt))
# #             else:
# #                 pass
# #         if b'\xFF\xD8' not in complete_sci[0]["data"]:
# #             log.error("Image initialization bytes FF D8 not found in first packet.")
# #         if b'\xFF\xD9' not in complete_sci[-1]["data"]:
# #             log.error("Image terminator bytes FF D9 not found in last packet.")
# #
# #         return cls(complete_sci)
# #
# #     @property
# #     def data(self) -> bytes:
# #         return self["data"].sum()
# #
# #
# # class JpegImage(ImagePackets):
# #     def __init__(self, *args, **kwargs) -> None:
# #         super().__init__(*args, **kwargs)
# #         n_starts = self.data.count(b'\xFF\xD8')
# #         n_endings = self.data.count(b'\xFF\xD9')
# #         self.n_images = min((n_starts, n_endings))
# #         if n_starts != n_endings:
# #             log.error(f"Number of JPEG initializations ({n_starts}) does not match "
# #                       f"number of terminations ({n_endings}).")
# #         log.debug(f"Initialized {type(self).__name__} object containing "
# #                   f"{self.n_images} JPEG image(s) in {len(self)} packets")
# #
# #     @property
# #     def images(self) -> t.List[bytes]:
# #         image_list = []
# #         bytes_left = self.data
# #         for i in range(self.n_images):
# #             d8_index = bytes_left.index(b'\xFF\xD8') + 1
# #             d9_index = bytes_left.index(b'\xFF\xD9') + 1
# #             image_list.append(bytes_left[d8_index - 1:d9_index + 1])
# #             bytes_left = bytes_left[d9_index + 1:]
# #
# #         return image_list
# #
# #     def save(self, filename):
# #         imgs = self.images
# #         for i in range(self.n_images):
# #             if self.n_images == 1:
# #                 fname_i = filename
# #             else:
# #                 base, ext = os.path.splitext(filename)
# #                 fname_i = f"{base}_{i}{ext}"
# #             with open(fname_i, "w+b") as jpegfile:
# #                 jpegfile.write(imgs[i])
# #
#
#
# def load_sci_packet_definitions(definitions_excel_path: str,
# packet_names: t.List[str]) -> t.Dict[str, t.Union[ccsdspy.VariableLength, ccsdspy.FixedLength]]:
#     definitions = {}
#     for packet_name in packet_names:
#         table = pd.read_excel(definitions_excel_path, sheet_name=packet_name)
#
#         packet_definition = []
#         for row in table.iterrows():
#             name = row[1][0]
#             kind = "uint"
#             size = row[1][8]
#             packet_definition.append(ccsdspy.PacketField(name=name, data_type=kind, bit_length=size))
#
#         # SCI_XFI packets have to be handled a bit special since they have an expanding field
#         if packet_name == "SCI_XFI":
#             packet_definition.pop()
#             packet_definition.append(ccsdspy.PacketArray(
#                 name="data",
#                 data_type="uint",
#                 bit_length=8,
#                 array_shape="expand",  # makes the data field expand
#             ))
#             packet_definition = ccsdspy.VariableLength(packet_definition[7:])  # 7: because we drop the primary header
#         else:
#             packet_definition = ccsdspy.FixedLength(packet_definition[7:])  # 7: because we drop the primary header
#         definitions[packet_name] = packet_definition
#     return definitions
#
#
# def unpack_ccsds(tlm_data_path: str,
#                  packet_definitions: t.Dict[str,
#                  t.Union[ccsdspy.VariableLength,
#                  ccsdspy.FixedLength]]) -> t.Dict[str, t.Dict[str, np.ndarray]]:
#     with open(tlm_data_path, "rb") as mixed_file:
#         stream_by_apid = ccsdspy.split_by_apid(mixed_file)
#
#     unpacked_contents = dict()
#     for apid, stream in stream_by_apid.items():
#         packet_name = APID_TO_PACKET_NAME[apid]
#         unpacked_contents[packet_name] = packet_definitions[packet_name].load(stream, include_primary_header=True)
#
#     return unpacked_contents

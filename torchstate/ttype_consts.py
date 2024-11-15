from enum import Enum

class ScalarTransferType(Enum):
    STR = 0
    INT64 = 1
    FLOAT64 = 2
    BOOL8 = 3

class TransferType(Enum):
    FLOAT32 = 4
    BFLOAT16 = 5
    FLOAT16 = 6
    UNIFORM_INT8 = 7

TTYPE_TO_ELEMENT_SIZE = {
    TransferType.FLOAT32.value: 4,
    TransferType.BFLOAT16.value: 2,
    TransferType.FLOAT16.value: 2,
    TransferType.UNIFORM_INT8.value: 1,
}

TTYPE_TO_CODEBOOK_SIZE = {
    TransferType.UNIFORM_INT8.value: 256,
}

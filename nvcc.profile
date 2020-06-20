TOP              = $(_HERE_)/..

PATH            += $(TOP)/@lib@/open64/bin:$(TOP)/@lib@/nvvm:

INCLUDES        +=  $(_SPACE_)

LIBRARIES        =+ $(_SPACE_) "-L$(TOP)/lib$(_TARGET_SIZE_)" -lcudart

CUDAFE_FLAGS    +=
OPENCC_FLAGS    +=
PTXAS_FLAGS     +=

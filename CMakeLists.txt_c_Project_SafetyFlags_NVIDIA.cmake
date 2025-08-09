# Safety flags for NVIDIA nvcc.
# Some of the safety flags may request certain optimization level.

list(APPEND SAFETY_FLAGS
    #--verbose
    #--display-error-number
    #--Werror=cross-execution-space-call
    #--Werror=reorder
    #--Werror=default-stream-launch
    #--Werror=ext-lambda-captures-this
    #--Werror=deprecated-declarations
    #--device-stack-protector=true
)

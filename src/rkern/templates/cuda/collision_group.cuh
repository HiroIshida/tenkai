#ifdef __CUDACC__
    #pragma unroll 1
#endif
    for (size_t i = 0; i < environment.@FIELD@.count; ++i) {
        const @NAMESPACE@::@PRIMITIVE@& obstacle = scene.@FIELD@[environment.@FIELD@.offset + i];
@BODY@        if (collision) return true;
    }

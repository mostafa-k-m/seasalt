set dotenv-load

build:
    CFLAGS="$ADD_CFLAGS $CFLAGS" && cythonize -i ./seasalt/cython_seasalt.pyx
    
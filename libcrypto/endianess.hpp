// ethashprime: C/C++ implementation of Ethashprime, the Ethereum Proof of Work algorithm.
// Copyright 2018-2019 Pawel Bylica.
// Licensed under the Apache License, Version 2.0.

// Modified by Firominer's authors 2021

#pragma once
#ifndef CRYPTO_ENDIANESS_HPP_
#define CRYPTO_ENDIANESS_HPP_

#if defined(_MSC_VER) || defined(_WIN32)

#include <stdlib.h>
#define bswap32 _byteswap_ulong
#define bswap64 _byteswap_uint64

// On Windows assume little endian.
#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN 4321
#define __BYTE_ORDER __LITTLE_ENDIAN

#elif __APPLE__

#include <machine/endian.h>

#define bswap32 __builtin_bswap32
#define bswap64 __builtin_bswap64

#else

#include <endian.h>

#define bswap32 __builtin_bswap32
#define bswap64 __builtin_bswap64

#endif

#endif  // !CRYPTO_ENDIANESS_HPP_
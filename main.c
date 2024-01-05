#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#define NUM_U32_PER_REG (sizeof(uint32x4_t) / sizeof(uint32_t))
# ifdef __APPLE__
// This solution obtained from:
// http://main.lv/writeup/arm64_assembly_crc32.md
#define CRC32X(crc, value) __asm__("crc32x %w[c], %w[c], %x[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC32W(crc, value) __asm__("crc32w %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC32H(crc, value) __asm__("crc32h %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC32B(crc, value) __asm__("crc32b %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC32CX(crc, value) __asm__("crc32cx %w[c], %w[c], %x[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC32CW(crc, value) __asm__("crc32cw %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC32CH(crc, value) __asm__("crc32ch %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC32CB(crc, value) __asm__("crc32cb %w[c], %w[c], %w[v]":[c]"+r"(crc):[v]"r"(value))
#define CRC_32_POLY 0xEDB88320
# else
#define CRC_32_POLY 0x04C11DB7
# endif
#elif defined(__SSE4_2__)
#include <immintrin.h>
#define CRC_32_POLY 0x1EDC6F41
#define NUM_U32_PER_M128 (sizeof(__mm128i) / sizeof(uint32_t))
# ifdef __AVX512__
#define NUM_U32_PER_M512 (sizeof(__mm512i) / sizeof(uint32_t))
# endif
#else
#error SIMD support required.
#endif

#define NUM_INT8       (UINT8_MAX + 1)
#define BYTE_BIT_WIDTH 8
#define NS_PER_S       1000000000

uint32_t crc32Table[NUM_INT8];

void makeCrc32TableSimd(uint32_t table[const static NUM_INT8])
{
#ifdef __AVX512F__
    const __m512i poly_vec = _mm512_set1_epi32(CRC_32_POLY);
    const __m512i one_vec = _mm512_set1_epi32(0x00000001);
    const __m512i inc_vec = _mm512_set1_epi32(NUM_U32_PER_M512);
    __m512i n_vec = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (uint32_t i = 0; (i + NUM_U32_PER_M512) <= NUM_INT8; i += NUM_U32_PER_M512) {
        __m512i c_vec = n_vec;
        for (int b = 0; b < BYTE_BIT_WIDTH; b++) {
            const __m512i csr_vec = _mm512_srli_epi32(c_vec, 1);
            const __m512i and_vec = _mm512_and_epi32(c_vec, one_vec);
            const __mmask16 mask = _mm512_cmpeq_epi32_mask(and_vec, one_vec);
            c_vec = _mm512_mask_xor_epi32(csr_vec, mask, poly_vec, csr_vec);
        }
        _mm512_storeu_epi32(table + i, c_vec);
        n_vec = _mm512_add_epi32(n_vec, inc_vec);
    }
#elif defined(__SSE4_2__)
    const __m128i poly_vec = _mm_set1_epi32(CRC_32_POLY);
    const __m128i one_vec = _mm_set1_epi32(0x00000001);
    const __m128i all_vec = _mm_set1_epi32(0xFFFFFFFF);
    const __m128i inc_vec = _mm_set1_epi32(NUM_U32_PER_M128);
    __m128i n_vec = _mm_set_epi32(3, 2, 1, 0);
    for (uint32_t i = 0; (i + NUM_U32_PER_M128) <= NUM_INT8; i += NUM_U32_PER_M128) {
        __m128i c_vec = n_vec;
        for (int b = 0; b < BYTE_BIT_WIDTH; b++) {
            const __m128i csr_vec = _mm_srli_epi32(c_vec, 1);
            const __m128i xor_vec = _mm_xor_si128(poly_vec, csr_vec);
            const __m128i and_vec = _mm_and_si128(c_vec, one_vec);
            const __m128i kept_xor_mask_vec = _mm_cmpeq_epi32(and_vec, one_vec);
            const __m128i kept_csr_mask_vec = _mm_xor_si128(kept_xor_mask_vec, all_vec);
            const __m128i kept_xor_vec = _mm_and_si128(xor_vec, kept_xor_mask_vec);
            const __m128i kept_csr_vec = _mm_and_si128(csr_vec, kept_csr_mask_vec);
            c_vec = _mm_or_si128(kept_csr_vec, kept_xor_vec);
        }
        _mm_storeu_si128((__m128i*)(table + i), c_vec);
        n_vec = _mm_add_epi32(n_vec, inc_vec);
    }
#elif defined(__ARM_NEON__)
    const uint32x4_t poly_vec = vdupq_n_u32(CRC_32_POLY);
    const uint32x4_t one_vec = vdupq_n_u32(0x00000001);
    const uint32x4_t inc_vec = vdupq_n_u32(NUM_U32_PER_REG);
    static const uint32_t nVecStartValues[NUM_U32_PER_REG] = {0, 1, 2, 3};
    uint32x4_t n_vec = vld1q_u32(nVecStartValues);
    for (uint32_t i = 0; (i + NUM_U32_PER_REG) <= NUM_INT8; i += NUM_U32_PER_REG) {
        uint32x4_t c_vec = n_vec;
        for (int b = 0; b < BYTE_BIT_WIDTH; b++) {
            const uint32x4_t csr_vec = vshrq_n_u32(c_vec, 1);
            const uint32x4_t xor_vec = veorq_u32(poly_vec, csr_vec);
            const uint32x4_t and_vec = vandq_u32(c_vec, one_vec);
            const uint32x4_t kept_xor_mask = vceqq_u32(and_vec, one_vec);
            const uint32x4_t kept_csr_mask = vmvnq_u32(kept_xor_mask);
            const uint32x4_t kept_xor = vandq_u32(xor_vec, kept_xor_mask);
            const uint32x4_t kept_csr = vandq_u32(csr_vec, kept_csr_mask);
            c_vec = vorrq_u32(kept_xor, kept_csr);
        }
        vst1q_u32(table + i, c_vec);
        n_vec = vaddq_u32(n_vec, inc_vec);
    }
#else
#error SIMD support required.
#endif
}

void printCrc32Table(const uint32_t table[const static NUM_INT8])
{
    int row_counter = 0;
    for (size_t i = 0; i < NUM_INT8; i++) {
        printf("%08X ", table[i]);
        row_counter++;
        if (row_counter == 8) {
            printf("\n");
            row_counter = 0;
        }
    }
}

uint32_t calculateCrc32(const size_t bufLen, const uint8_t buffer[const bufLen])
{
    uint32_t crc32 = UINT32_MAX;
    for (size_t i = 0; i < bufLen; i++) {
        crc32 ^= buffer[i];
        for (size_t bit = 0; bit < BYTE_BIT_WIDTH; bit++) {
            if (crc32 & 0x00000001) {
                crc32 = CRC_32_POLY ^ (crc32 >> 1);
            } else {
                crc32 = crc32 >> 1;
            }
        }
    }
    return ~crc32;
}

uint32_t calculateCrc32ViaTable(const size_t bufLen, const uint8_t buffer[const bufLen])
{
    uint32_t crc32 = UINT32_MAX;
    for (size_t i = 0; i < bufLen; i++) {
        const uint32_t index = (crc32 ^ buffer[i]) & UINT8_MAX;
        crc32 = crc32Table[index] ^ (crc32 >> BYTE_BIT_WIDTH);
    }
    return crc32 ^ UINT32_MAX;
}

uint32_t calculateCrc32Simd(const size_t bufLen, const uint8_t buffer[const bufLen])
{
    uint32_t crc32 = UINT32_MAX;
    for (size_t i = 0; i < bufLen; i++) {
#ifdef __SSE4_2__
        crc32 = _mm_crc32_u8(crc32, buffer[i]);
#elif defined(__APPLE__)
        CRC32B(crc32, buffer[i]);
#elif defined(__ARM_FEATURE_CRC32)
        crc32 = __crc32cb(crc32, buffer[i]);
#else
#error SIMD support required.
#endif
    }
    return ~crc32;
}

int64_t getTimeDiffNs(const struct timespec* const start, const struct timespec* const stop)
{
    int64_t sec = stop->tv_sec - start->tv_sec;
    int64_t nsec = stop->tv_nsec - start->tv_nsec;
    if ((sec < 0) && (nsec > 0)) {
        sec++;
        nsec -= NS_PER_S;
    } else if ((sec > 0) && (nsec < 0)) {
        sec--;
        nsec += NS_PER_S;
    }
    return (sec * NS_PER_S) + nsec;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Error: This program requires 1 (and only 1) argument.\n");
        return EXIT_FAILURE;
    }

    const size_t len = strlen(argv[1]);
    if (len < 1) {
        fprintf(stderr, "Error: Invalid input parameter.\n");
        return EXIT_FAILURE;
    }

    static uint32_t (* const crc32Funcs[3])(const size_t, const uint8_t* const) = {
        &calculateCrc32,
        &calculateCrc32ViaTable,
        &calculateCrc32Simd
    };
    static const char* const crc32FuncNames[3] = {
        "SISD",
        "Table",
        "SIMD"
    };
    static const int nameColWidth = 5;

    makeCrc32TableSimd(crc32Table);
    //printCrc32Table(crc32Table);
    //printf("\n");

    printf("%-*s | %-*s | Time (ns)\n", nameColWidth, "Name", 10, "Result");
    for (int i = 0; i < 3; i++) {
        struct timespec start;
        clock_gettime(CLOCK_MONOTONIC, &start);
        const uint32_t crc32 = crc32Funcs[i](len, (uint8_t*)argv[1]);
        struct timespec stop;
        clock_gettime(CLOCK_MONOTONIC, &stop);
        const int64_t timeElapsedNs = getTimeDiffNs(&start, &stop);
        printf("%-*s | 0x%08X | %" PRIi64 "\n", nameColWidth, crc32FuncNames[i], crc32, timeElapsedNs);
    }

    return 0;
}


// progpowprime: C/C++ implementation of Progpowprime.
// Copyright 2018-2019 Pawel Bylica.
// Licensed under the Apache License, Version 2.0.

// Modified by Firominer's authors 2021

#include "progpowprime.hpp"
#include "bitwise.hpp"

namespace progpowprime
{
mix_rng_state::mix_rng_state(uint64_t seed) noexcept
{
    const auto seed_lo{static_cast<uint32_t>(seed)};
    const auto seed_hi{static_cast<uint32_t>(seed >> 32)};

    const auto z{crypto::fnv1a(crypto::kFNV_OFFSET_BASIS, seed_lo)};
    const auto w{crypto::fnv1a(z, seed_hi)};
    const auto jsr{crypto::fnv1a(w, seed_lo)};
    const auto jcong{crypto::fnv1a(jsr, seed_hi)};

    rng = crypto::kiss99{z, w, jsr, jcong};

    // Create random permutations of mix destinations / sources.
    // Uses Fisher-Yates shuffle.
    for (uint32_t i{0}; i < kRegs; ++i)
    {
        dst_seq_[i] = i;
        src_seq_[i] = i;
    }

    for (uint32_t i{kRegs}; i > 1; --i)
    {
        std::swap(dst_seq_[i - 1], dst_seq_[rng() % i]);
        std::swap(src_seq_[i - 1], src_seq_[rng() % i]);
    }
}

NO_SANITIZE("unsigned-integer-overflow")
static void random_merge(uint32_t& a, uint32_t b, uint32_t sel) noexcept
{
    const auto x = (sel >> 16) % 31 + 1;  // Additional non-zero selector from higher bits.
    switch (sel % 4)
    {
    case 0:
        a = (a * 33) + b;
        return;
    case 1:
        a = (a ^ b) * 33;
        return;
    case 2:
        a = crypto::rotl32(a, x) ^ b;
        return;
    case 3:
        a = crypto::rotr32(a, x) ^ b;
        return;
    };
}

// Merge new data from b into the value in a
// Assuming A has high entropy only do ops that retain entropy, even if B is low entropy
// (IE don't do A&B)
static std::string random_merge_src(std::string a, std::string b, uint32_t r)
{
    const auto x{((r >> 16) % 31) + 1};  // Additional non-zero selector from higher bits.

    switch (r % 4)
    {
    case 0:
        return a + " = (" + a + " * 33) + " + b + ";\n";
    case 1:
        return a + " = (" + a + " ^ " + b + ") * 33;\n";
    case 2:
        return a + " = ROTL32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";\n";
    case 3:
        return a + " = ROTR32(" + a + ", " + std::to_string(x) + ") ^ " + b + ";\n";
    }
    return "#error\n";
}

NO_SANITIZE("unsigned-integer-overflow")
static uint32_t random_math(uint32_t a, uint32_t b, uint32_t sel) noexcept
{
    switch (sel % 11)
    {
    case 0:
        return a + b;
    case 1:
        return a * b;
    case 2:
        return crypto::mul_hi32(a, b);
    case 3:
        return std::min(a, b);
    case 4:
        return crypto::rotl32(a, b);
    case 5:
        return crypto::rotr32(a, b);
    case 6:
        return a & b;
    case 7:
        return a | b;
    case 8:
        return a ^ b;
    case 9:
        return crypto::clz32(a) + crypto::clz32(b);
    default: /* 10 */
        return crypto::popcnt32(a) + crypto::popcnt32(b);
    }
}

// Random math between two input values
static std::string random_math_src(std::string d, std::string a, std::string b, uint32_t r)
{
    switch (r % 11)
    {
    case 0:
        return d + " = " + a + " + " + b + ";\n";
    case 1:
        return d + " = " + a + " * " + b + ";\n";
    case 2:
        return d + " = mul_hi(" + a + ", " + b + ");\n";
    case 3:
        return d + " = min(" + a + ", " + b + ");\n";
    case 4:
        return d + " = ROTL32(" + a + ", " + b + " % 32);\n";
    case 5:
        return d + " = ROTR32(" + a + ", " + b + " % 32);\n";
    case 6:
        return d + " = " + a + " & " + b + ";\n";
    case 7:
        return d + " = " + a + " | " + b + ";\n";
    case 8:
        return d + " = " + a + " ^ " + b + ";\n";
    case 9:
        return d + " = clz(" + a + ") + clz(" + b + ");\n";
    case 10:
        return d + " = popcount(" + a + ") + popcount(" + b + ");\n";
    }
    return "#error\n";
}

std::string getKern(uint64_t prog_seed, kernel_type kern)
{
    std::stringstream ret;
    mix_rng_state state{prog_seed};

    if (kern == kernel_type::Cuda)
    {
        ret << "typedef unsigned int       uint32_t;\n";
        ret << "typedef unsigned long long uint64_t;\n";
        ret << "#if __CUDA_ARCH__ < 350\n";
        ret << "#define ROTL32(x,n) (((x) << (n % 32)) | ((x) >> (32 - (n % 32))))\n";
        ret << "#define ROTR32(x,n) (((x) >> (n % 32)) | ((x) << (32 - (n % 32))))\n";
        ret << "#else\n";
        ret << "#define ROTL32(x,n) __funnelshift_l((x), (x), (n))\n";
        ret << "#define ROTR32(x,n) __funnelshift_r((x), (x), (n))\n";
        ret << "#endif\n";
        ret << "#define min(a,b) ((a<b) ? a : b)\n";
        ret << "#define mul_hi(a, b) __umulhi(a, b)\n";
        ret << "#define clz(a) __clz(a)\n";
        ret << "#define popcount(a) __popc(a)\n\n";

        ret << "#define DEV_INLINE __device__ __forceinline__\n";
        ret << "#if (__CUDACC_VER_MAJOR__ > 8)\n";
        ret << "#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))\n";
        ret << "#else\n";
        ret << "#define SHFL(x, y, z) __shfl((x), (y), (z))\n";
        ret << "#endif\n\n";

        ret << "\n";
    }
    else
    {
        ret << "#ifndef GROUP_SIZE\n";
        ret << "#define GROUP_SIZE 128\n";
        ret << "#endif\n";
        ret << "#define GROUP_SHARE (GROUP_SIZE / " << kLanes << ")\n";
        ret << "\n";
        ret << "typedef unsigned int       uint32_t;\n";
        ret << "typedef unsigned long      uint64_t;\n";
        ret << "#define ROTL32(x, n) rotate((x), (uint32_t)(n))\n";
        ret << "#define ROTR32(x, n) rotate((x), (uint32_t)(32-n))\n";
        ret << "\n";
    }

    ret << "#define PROGPOWPRIME_LANES           " << kLanes << "\n";
    ret << "#define PROGPOWPRIME_REGS            " << kRegs << "\n";
    ret << "#define PROGPOWPRIME_DAG_LOADS       " << kDag_loads << "\n";
    ret << "#define PROGPOWPRIME_CACHE_WORDS     " << kCache_bytes / sizeof(uint32_t) << "\n";
    ret << "#define PROGPOWPRIME_CNT_DAG         " << kDag_count << "\n";
    ret << "#define PROGPOWPRIME_CNT_MATH        " << kMath_count << "\n";
    ret << "\n";

    if (kern == kernel_type::Cuda)
    {
        ret << "typedef struct __align__(16) {uint32_t s[PROGPOWPRIME_DAG_LOADS];} dag_t;\n";
        ret << "\n";
        ret << "// Inner loop for prog_seed " << prog_seed << "\n";
        ret << "__device__ __forceinline__ void progpowprimeLoop(const uint32_t loop,\n";
        ret << "        uint32_t mix[PROGPOWPRIME_REGS],\n";
        ret << "        const dag_t *g_dag,\n";
        ret << "        const uint32_t c_dag[PROGPOWPRIME_CACHE_WORDS],\n";
        ret << "        const bool hack_false)\n";
    }
    else
    {
        ret << "typedef struct __attribute__ ((aligned (16))) {uint32_t s[PROGPOWPRIME_DAG_LOADS];} "
               "dag_t;\n";
        ret << "\n";
        ret << "// Inner loop for prog_seed " << prog_seed << "\n";
        ret << "inline void progpowprimeLoop(const uint32_t loop,\n";
        ret << "        volatile uint32_t mix_arg[PROGPOWPRIME_REGS],\n";
        ret << "        __global const dag_t *g_dag,\n";
        ret << "        __local const uint32_t c_dag[PROGPOWPRIME_CACHE_WORDS],\n";
        ret << "        __local uint64_t share[GROUP_SHARE],\n";
        ret << "        const bool hack_false)\n";
    }
    ret << "{\n";

    ret << "dag_t data_dag;\n";
    ret << "uint32_t offset, data;\n";
    // Work around AMD OpenCL compiler bug
    // See https://github.com/gangnamtestnet
    if (kern == kernel_type::OpenCL)
    {
        ret << "uint32_t mix[PROGPOWPRIME_REGS];\n";
        ret << "for(int i=0; i<PROGPOWPRIME_REGS; i++)\n";
        ret << "    mix[i] = mix_arg[i];\n";
    }

    if (kern == kernel_type::Cuda)
        ret << "const uint32_t lane_id = threadIdx.x & (PROGPOWPRIME_LANES-1);\n";
    else
    {
        ret << "const uint32_t lane_id = get_local_id(0) & (PROGPOWPRIME_LANES-1);\n";
        ret << "const uint32_t group_id = get_local_id(0) / PROGPOWPRIME_LANES;\n";
    }

    // Global memory access
    // lanes access sequential locations
    // Hard code mix[0] to guarantee the address for the global load depends on the result of the
    // load
    ret << "// global load\n";
    if (kern == kernel_type::Cuda)
        ret << "offset = SHFL(mix[0], loop%PROGPOWPRIME_LANES, PROGPOWPRIME_LANES);\n";
    else
    {
        ret << "if(lane_id == (loop % PROGPOWPRIME_LANES))\n";
        ret << "    share[group_id] = mix[0];\n";
        ret << "barrier(CLK_LOCAL_MEM_FENCE);\n";
        ret << "offset = share[group_id];\n";
    }
    ret << "offset %= PROGPOWPRIME_DAG_ELEMENTS;\n";
    ret << "offset = offset * PROGPOWPRIME_LANES + (lane_id ^ loop) % PROGPOWPRIME_LANES;\n";
    ret << "data_dag = g_dag[offset];\n";
    ret << "// hack to prevent compiler from reordering LD and usage\n";
    if (kern == kernel_type::Cuda)
        ret << "if (hack_false) __threadfence_block();\n";
    else
        ret << "if (hack_false) barrier(CLK_LOCAL_MEM_FENCE);\n";

    for (int i = 0; (i < kCache_count) || (i < kMath_count); i++)
    {
        if (i < kCache_count)
        {
            // Cached memory access
            // lanes access random locations
            std::string src{"mix[" + std::to_string(state.next_src()) + "]"};
            std::string dest{"mix[" + std::to_string(state.next_dst()) + "]"};
            const auto sel{state.rng()};

            ret << "// cache load " << i << "\n";
            ret << "offset = " << src << " % PROGPOWPRIME_CACHE_WORDS;\n";
            ret << "data = c_dag[offset];\n";
            ret << random_merge_src(dest, "data", sel);
        }
        if (i < kMath_count)
        {
            // Random Math
            // Generate 2 unique sources
            auto src_rnd{state.rng() % (kRegs * (kRegs - 1))};
            auto src1{src_rnd % kRegs};  // 0 <= src1 < kRegs
            auto src2{src_rnd / kRegs};  // 0 <= src2 < kRegs - 1
            if (src2 >= src1)
            {
                ++src2;  // src2 is now any reg other than src1
            }

            std::string src1_str = "mix[" + std::to_string(src1) + "]";
            std::string src2_str = "mix[" + std::to_string(src2) + "]";

            auto sel1{state.rng()};
            auto sel2{state.rng()};

            std::string dest{"mix[" + std::to_string(state.next_dst()) + "]"};

            ret << "// random math " << i << "\n";
            ret << random_math_src("data", src1_str, src2_str, sel1);
            ret << random_merge_src(dest, "data", sel2);
        }
    }
    // Consume the global load data at the very end of the loop, to allow fully latency hiding
    ret << "// consume global load data\n";
    ret << "// hack to prevent compiler from reordering LD and usage\n";
    if (kern == kernel_type::Cuda)
    {
        ret << "if (hack_false) __threadfence_block();\n";
    }
    else
    {
        ret << "if (hack_false) barrier(CLK_LOCAL_MEM_FENCE);\n";
    }

    ret << random_merge_src("mix[0]", "data_dag.s[0]", state.rng());
    for (int i = 1; i < kDag_loads; i++)
    {
        std::string dst{"mix[" + std::to_string(state.next_dst()) + "]"};
        std::string src{"data_dag.s[" + std::to_string(i) + "]"};
        ret << random_merge_src(dst, src, state.rng());
    }

    // Work around AMD OpenCL compiler bug
    if (kern == kernel_type::OpenCL)
    {
        ret << "for(int i=0; i<PROGPOWPRIME_REGS; i++)\n";
        ret << "    mix_arg[i] = mix[i];\n";
    }
    ret << "}\n";
    ret << "\n";

    return ret.str();
}

using mix_t = std::array<std::array<uint32_t, kRegs>, kLanes>;

static void round(const ethashprime::epoch_context& context, uint32_t r, mix_t& mix, mix_rng_state state)
{
    static const uint32_t l1_cache_words{ethashprime::kL1_cache_size / sizeof(uint32_t)};
    const uint32_t num_items{static_cast<uint32_t>(context.full_dataset_num_items / 2)};
    const uint32_t item_index{mix.at(r % kLanes).at(0) % num_items};

    // Load DAG Data
    ethashprime::hash2048 item{ethashprime::detail::lazy_lookup_2048(context, item_index)};

    const auto max_operations{std::max(kCache_count, kMath_count)};

    // Process lanes.
    for (unsigned i = 0; i < max_operations; ++i)
    {
        if (i < kCache_count)  // Random access to cached memory.
        {
            const auto src{state.next_src()};
            const auto dst{state.next_dst()};
            const auto sel{state.rng()};

            for (uint64_t l{0}; l < kLanes; ++l)
            {
                const size_t offset = mix.at(l).at(src) % ethashprime::kL1_cache_words;
                random_merge(mix.at(l).at(dst), ethashprime::le::uint32(context.l1_cache[offset]), sel);
            }
        }
        if (i < kMath_count)  // Random math.
        {
            // Generate 2 unique source indexes.
            const auto src_rnd{state.rng() % (kRegs * (kRegs - 1))};
            const auto src1{src_rnd % kRegs};  // O <= src1 < num_regs
            auto src2{src_rnd / kRegs};        // 0 <= src2 < num_regs - 1
            if (src2 >= src1)
            {
                ++src2;
            }

            const auto sel1{state.rng()};
            const auto dst{state.next_dst()};
            const auto sel2{state.rng()};

            for (uint64_t l{0}; l < kLanes; ++l)
            {
                const uint32_t data = random_math(mix.at(l).at(src1), mix.at(l).at(src2), sel1);
                random_merge(mix.at(l).at(dst), data, sel2);
            }
        }
    }

    // Dag Access pattern
    uint32_t dsts[kWords_per_lane];
    uint32_t sels[kWords_per_lane];
    for (size_t i = 0; i < kWords_per_lane; i++)
    {
        dsts[i] = (i == 0 ? 0 : state.next_dst());
        sels[i] = state.rng();
    }

    // Dag access
    for (size_t l = 0; l < kLanes; l++)
    {
        const auto offset = ((l ^ r) % kLanes) * kWords_per_lane;
        for (size_t i = 0; i < kWords_per_lane; i++)
        {
            const auto word = ethashprime::le::uint32(item.word32s[offset + i]);
            random_merge(mix.at(l).at(dsts[i]), word, sels[i]);
        }
    }
}


static mix_t init_mix(uint64_t seed)
{
    const uint32_t z = crypto::fnv1a(crypto::kFNV_OFFSET_BASIS, static_cast<uint32_t>(seed));
    const uint32_t w = crypto::fnv1a(z, static_cast<uint32_t>(seed >> 32));

    mix_t mix;
    for (uint32_t l{0}; l < mix.size(); l++)
    {
        const uint32_t jsr = crypto::fnv1a(w, l);
        const uint32_t jcong = crypto::fnv1a(jsr, l);
        crypto::kiss99 rng{z, w, jsr, jcong};
        for (auto& row : mix.at(l))
        {
            row = rng();
        }
    }
    return mix;
}

static const uint32_t meowcoin_meowpow[15] = {
        0x0000004D, //M
        0x00000045, //E
        0x0000004F, //O
        0x00000057, //W
        0x00000043, //C
        0x0000004F, //O
        0x00000049, //I
        0x0000004E, //N
        0x0000004D, //M
        0x00000045, //E
        0x0000004F, //O
        0x00000057, //W
        0x00000050, //P
        0x0000004F, //O
        0x00000057, //W
};

ethashprime::hash256 hash_seed(const ethashprime::hash256& header_hash, uint64_t nonce) noexcept
{
    nonce = ethashprime::le::uint64(nonce);
    uint32_t state[25] = {0x0};
    for (size_t i = 0; i < 8; i++)
    {
        state[i] = ethashprime::le::uint32(header_hash.word32s[i]);
    }

    std::memcpy(&state[8], &nonce, sizeof(uint64_t));
//    state[10] = 0x00000001;
//    state[18] = 0x80008081;
      // 3rd apply meowcoin input constraints
      for (int i = 10; i < 25; i++)
          state[i] = meowcoin_meowpow[i-10];

    ethashprime::keccakf800(state);

    ethashprime::hash256 output;
    for (int i = 0; i < 8; ++i)
    {
        output.word32s[i] = ethashprime::le::uint32(state[i]);
    }
    return output;
}

ethashprime::hash256 hash_mix(const ethashprime::epoch_context& context, const uint32_t period, uint64_t seed)
{
    auto mix{init_mix(seed)};
    mix_rng_state state(period);

    for (uint32_t i{0}; i < kDag_count; ++i)
    {
        round(context, i, mix, state);
    }

    // Reduce mix data to a single per-lane result.
    uint32_t lane_hash[kLanes];
    for (size_t l{0}; l < kLanes; ++l)
    {
        lane_hash[l] = crypto::kFNV_OFFSET_BASIS;
        for (uint32_t i{0}; i < kRegs; ++i)
        {
            lane_hash[l] = crypto::fnv1a(lane_hash[l], mix.at(l).at(i));
        }
    }

    // Reduce all lanes to a single 256-bit result.
    static const size_t num_words{sizeof(ethashprime::hash256) / sizeof(uint32_t)};
    ethashprime::hash256 mix_hash{};
    for (auto& w : mix_hash.word32s)
    {
        w = crypto::kFNV_OFFSET_BASIS;
    }

    for (size_t l{0}; l < kLanes; ++l)
    {
        mix_hash.word32s[l % num_words] = crypto::fnv1a(mix_hash.word32s[l % num_words], lane_hash[l]);
    }

#if __BYTE_ORDER != __LITTLE_ENDIAN
    for (auto& w : mix_hash.word32s)
    {
        w = ethashprime::le::uint32(w);
    }
#endif

    return mix_hash;
}

ethashprime::hash256 hash_final(const ethashprime::hash256& input_hash, const ethashprime::hash256& mix_hash) noexcept
{
    uint32_t state[25] = {0};
    std::memcpy(&state[0], input_hash.bytes, sizeof(ethashprime::hash256));
    std::memcpy(&state[8], mix_hash.bytes, sizeof(ethashprime::hash256));
    //state[17] = 0x00000001;
    //state[24] = 0x80008081;
    // 3rd apply meowcoin input constraints
    for (int i = 16; i < 25; i++)
        state[i] = meowcoin_meowpow[i - 16];
    ethashprime::keccakf800(state);
    ethashprime::hash256 output{};
    std::memcpy(output.bytes, &state[0], sizeof(ethashprime::hash256));
    return output;
}

ethashprime::result hash(
    const ethashprime::epoch_context& context, const uint32_t period, const ethashprime::hash256& header_hash, uint64_t nonce)
{
    const ethashprime::hash256 seed_hash{progpowprime::hash_seed(header_hash, nonce)};
    const uint64_t seed_64{seed_hash.word64s[0]};
    const ethashprime::hash256 mix_hash{progpowprime::hash_mix(context, period, seed_64)};
    const ethashprime::hash256 final_hash{progpowprime::hash_final(seed_hash, mix_hash)};
    return {final_hash, mix_hash};
}

ethashprime::VerificationResult verify_full(const ethashprime::epoch_context& context, const uint32_t period,
    const ethashprime::hash256& header_hash, const ethashprime::hash256& mix_hash, uint64_t nonce,
    const ethashprime::hash256& boundary) noexcept
{
    auto result{progpowprime::hash(context, period, header_hash, nonce)};
    if (!ethashprime::is_less_or_equal(result.final_hash, boundary))
    {
        return ethashprime::VerificationResult::kInvalidNonce;
    }
    if (!ethashprime::is_equal(result.mix_hash, mix_hash))
    {
        return ethashprime::VerificationResult::kInvalidMixHash;
    }
    return ethashprime::VerificationResult::kOk;
}

ethashprime::VerificationResult verify_full(const uint64_t block_number, const ethashprime::hash256& header_hash,
    const ethashprime::hash256& mix_hash, uint64_t nonce, const ethashprime::hash256& boundary) noexcept
{
    auto dag_epoch_number{ethashprime::calculate_epoch_from_block_num(block_number)};
    auto dag_epoch_context{ethashprime::get_epoch_context(dag_epoch_number, false)};
    auto progpowprime_period{block_number / progpowprime::kPeriodLength};
    return progpowprime::verify_full(*dag_epoch_context, progpowprime_period, header_hash, mix_hash, nonce, boundary);
}


}  // namespace progpowprime

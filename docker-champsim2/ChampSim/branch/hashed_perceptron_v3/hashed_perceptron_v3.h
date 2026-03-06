#ifndef BRANCH_HASHED_PERCEPTRON_V3_H
#define BRANCH_HASHED_PERCEPTRON_V3_H

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "modules.h"
#include "msl/bits.h"
#include "msl/fwcounter.h"
#include "../hashed_perceptron/folded_shift_register.h"

class hashed_perceptron_v3 : champsim::modules::branch_predictor
{
  using bits = champsim::data::bits;

  constexpr static std::size_t GLOBAL_TABLES = 16;
  constexpr static std::size_t TOTAL_TABLES  = GLOBAL_TABLES + 2; // bias + local

  constexpr static std::size_t TABLE_SIZE = 1 << 12;
  constexpr static bits TABLE_INDEX_BITS{champsim::msl::lg2(TABLE_SIZE)};

  constexpr static std::size_t LOCAL_SIZE = 1 << 10; // 1024 local entries

  // 10-bit weights
  std::array<std::array<champsim::msl::sfwcounter<10>, TABLE_SIZE>, TOTAL_TABLES> tables{};

  // Global geometric history
  std::array<bits, GLOBAL_TABLES> history_lengths = {
      bits{4}, bits{8}, bits{12}, bits{16}, bits{24}, bits{32},
      bits{48}, bits{64}, bits{80}, bits{96}, bits{128},
      bits{160}, bits{192}, bits{224}, bits{256}, bits{320}
  };

  using history_type = folded_shift_register<TABLE_INDEX_BITS>;
  std::array<history_type, GLOBAL_TABLES> ghist_words = []() {
    decltype(ghist_words) arr;
    std::array<bits, GLOBAL_TABLES> lengths = {
      bits{4}, bits{8}, bits{12}, bits{16}, bits{24}, bits{32},
      bits{48}, bits{64}, bits{80}, bits{96}, bits{128},
      bits{160}, bits{192}, bits{224}, bits{256}, bits{320}
    };
    for (std::size_t i = 0; i < GLOBAL_TABLES; i++)
      arr[i] = history_type{lengths[i]};
    return arr;
  }();

  // Local history table
  std::array<uint16_t, LOCAL_SIZE> local_history{};

  int theta = 30;
  int tc = 0;

  struct result_t {
    std::array<uint64_t, TOTAL_TABLES> indices{};
    int yout = 0;
  };

  result_t last_result{};

public:
  using branch_predictor::branch_predictor;

  bool predict_branch(champsim::address pc);
  void last_branch_result(champsim::address pc,
                          champsim::address,
                          bool taken,
                          uint8_t);

  void adjust_threshold(bool correct);
};

#endif

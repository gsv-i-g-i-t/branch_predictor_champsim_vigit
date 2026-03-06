#ifndef BRANCH_HASHED_PERCEPTRON_PRO_H
#define BRANCH_HASHED_PERCEPTRON_PRO_H

#include <array>
#include <cstdint>
#include <numeric>

#include "modules.h"
#include "msl/bits.h"
#include "msl/fwcounter.h"
#include "../hashed_perceptron/folded_shift_register.h"

class hashed_perceptron_pro : champsim::modules::branch_predictor
{
  using bits = champsim::data::bits;

  constexpr static std::size_t NTABLES = 16;
  constexpr static std::size_t TABLE_SIZE = 1 << 13;
  constexpr static bits TABLE_INDEX_BITS{champsim::msl::lg2(TABLE_SIZE)};

  constexpr static std::size_t LONG_HISTORY_LEN = 512;

  std::array<std::array<champsim::msl::sfwcounter<8>, TABLE_SIZE>, NTABLES> tables{};

  constexpr static std::array<bits, NTABLES> history_lengths = {
      bits{3}, bits{6}, bits{10}, bits{16},
      bits{24}, bits{32}, bits{48}, bits{64},
      bits{96}, bits{128}, bits{160}, bits{192},
      bits{224}, bits{256}, bits{320}, bits{384}
  };

  using history_type = folded_shift_register<TABLE_INDEX_BITS>;

  std::array<history_type, NTABLES> ghist_words = []() {
    decltype(ghist_words) arr;
    for (std::size_t i = 0; i < NTABLES; i++)
      arr[i] = history_type{history_lengths[i]};
    return arr;
  }();

  // 🔥 Ultra-long reinforcement history
  history_type long_history = history_type{bits{LONG_HISTORY_LEN}};
  std::array<champsim::msl::sfwcounter<8>, TABLE_SIZE> long_table{};

  int theta = 10;
  int tc = 0;

  struct result_t {
    std::array<uint64_t, NTABLES> indices{};
    uint64_t long_index{};
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

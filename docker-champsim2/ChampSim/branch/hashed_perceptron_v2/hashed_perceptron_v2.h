#ifndef BRANCH_HASHED_PERCEPTRON_V2_H
#define BRANCH_HASHED_PERCEPTRON_V2_H

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>
#include <numeric>

#include "folded_shift_register.h"
#include "modules.h"
#include "msl/bits.h"
#include "msl/fwcounter.h"

class hashed_perceptron_v2 : champsim::modules::branch_predictor
{
  using bits = champsim::data::bits;

  constexpr static std::size_t NTABLES = 17;          // +1 bias table
  constexpr static bits MAXHIST{232};
  constexpr static bits MINHIST{3};

  constexpr static std::size_t TABLE_SIZE = 1 << 12;
  constexpr static bits TABLE_INDEX_BITS{champsim::msl::lg2(TABLE_SIZE)};

  constexpr static int THRESHOLD = 1;

  constexpr static std::array<bits, NTABLES-1> history_lengths = {
      MINHIST, bits{4}, bits{6}, bits{8}, bits{10}, bits{14}, bits{19},
      bits{26}, bits{36}, bits{49}, bits{67}, bits{91},
      bits{125}, bits{170}, bits{200}, MAXHIST
  };

  // 10-bit signed weights (wider range than original 8-bit)
  std::array<std::array<champsim::msl::sfwcounter<10>, TABLE_SIZE>, NTABLES> tables{};

  using history_type = folded_shift_register<TABLE_INDEX_BITS>;

  std::array<history_type, NTABLES-1> ghist_words = []() {
    decltype(ghist_words) retval;
    std::transform(std::cbegin(history_lengths), std::cend(history_lengths),
                   std::begin(retval),
                   [](const auto len) { return history_type{len}; });
    return retval;
  }();

  int theta = 20;  // slightly higher initial theta
  int tc = 0;

  struct perceptron_result {
    std::array<uint64_t, NTABLES> indices{};
    int yout = 0;
  };

  perceptron_result last_result{};

public:
  using branch_predictor::branch_predictor;

  bool predict_branch(champsim::address pc);
  void last_branch_result(champsim::address pc,
                          champsim::address branch_target,
                          bool taken,
                          uint8_t branch_type);
  void adjust_threshold(bool correct);
};

#endif

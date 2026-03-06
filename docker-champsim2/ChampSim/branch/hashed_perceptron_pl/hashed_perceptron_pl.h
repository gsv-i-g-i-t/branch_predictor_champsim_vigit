#ifndef BRANCH_HASHED_PERCEPTRON_PL_H
#define BRANCH_HASHED_PERCEPTRON_PL_H

#include <array>
#include <cstdint>
#include <numeric>

#include "modules.h"
#include "msl/bits.h"
#include "msl/fwcounter.h"
#include "../hashed_perceptron/folded_shift_register.h"

class hashed_perceptron_pl : champsim::modules::branch_predictor
{
  using bits = champsim::data::bits;

  constexpr static std::size_t NTABLES = 16;
  constexpr static std::size_t TABLE_SIZE = 1 << 12;
  constexpr static bits INDEX_BITS{champsim::msl::lg2(TABLE_SIZE)};

  // NEW: PC partitioning (piecewise dimension)
  constexpr static std::size_t PC_PARTITIONS = 8;

  std::array<
      std::array<
          std::array<champsim::msl::sfwcounter<8>, TABLE_SIZE>,
          PC_PARTITIONS>,
      NTABLES> tables{};

  std::array<bits, NTABLES> history_lengths = {
      bits{}, bits{4}, bits{8}, bits{12},
      bits{16}, bits{24}, bits{32}, bits{48},
      bits{64}, bits{96}, bits{128}, bits{160},
      bits{192}, bits{224}, bits{256}, bits{320}
  };

  using history_type = folded_shift_register<INDEX_BITS>;
  std::array<history_type, NTABLES> ghist_words;

  int theta = 12;
  int tc = 0;

  struct result_t {
    std::array<uint64_t, NTABLES> indices{};
    std::size_t pc_partition{};
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

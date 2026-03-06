#ifndef BRANCH_HASHED_PERCEPTRON_ELITE_H
#define BRANCH_HASHED_PERCEPTRON_ELITE_H

#include <array>
#include <cstdint>
#include <numeric>

#include "modules.h"
#include "msl/bits.h"
#include "msl/fwcounter.h"
#include "../hashed_perceptron/folded_shift_register.h"

class hashed_perceptron_elite : champsim::modules::branch_predictor
{
  using bits = champsim::data::bits;

  constexpr static std::size_t NTABLES = 16;
  constexpr static std::size_t TABLE_SIZE = 1 << 12;
  constexpr static bits INDEX_BITS{champsim::msl::lg2(TABLE_SIZE)};
  constexpr static std::size_t TAG_BITS = 8;

  struct entry {
    champsim::msl::sfwcounter<10> weight{};
    uint16_t tag = 0;
    uint8_t usefulness = 0;
  };

  std::array<std::array<entry, TABLE_SIZE>, NTABLES> tables{};

  std::array<bits, NTABLES> history_lengths = {
      bits{0}, bits{4}, bits{8}, bits{12},
      bits{16}, bits{24}, bits{32}, bits{48},
      bits{64}, bits{96}, bits{128}, bits{160},
      bits{192}, bits{224}, bits{256}, bits{320}
  };

  using history_type = folded_shift_register<INDEX_BITS>;
  std::array<history_type, NTABLES> ghist_words;

  int theta = 28;
  int tc = 0;

  struct result_t {
    std::array<uint64_t, NTABLES> indices{};
    std::array<bool, NTABLES> tag_match{};
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

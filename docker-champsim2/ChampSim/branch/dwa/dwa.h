#ifndef DWA_BRANCH_H
#define DWA_BRANCH_H

#include <array>
#include <bitset>

#include "modules.h"
#include "msl/fwcounter.h"

struct dwa : champsim::modules::branch_predictor {

  static constexpr std::size_t GLOBAL_HISTORY_LENGTH_SHORT = 6;
  static constexpr std::size_t GLOBAL_HISTORY_LENGTH_MEDIUM = 12;
  static constexpr std::size_t GLOBAL_HISTORY_LENGTH_LONG = 20;

  static constexpr std::size_t COUNTER_BITS = 2;
  static constexpr std::size_t GS_HISTORY_TABLE_SIZE = 16384;

  std::bitset<GLOBAL_HISTORY_LENGTH_SHORT> branch_history_vector_short{};
  std::bitset<GLOBAL_HISTORY_LENGTH_MEDIUM> branch_history_vector_medium{};
  std::bitset<GLOBAL_HISTORY_LENGTH_LONG> branch_history_vector_long{};

  std::array<champsim::msl::fwcounter<COUNTER_BITS>, GS_HISTORY_TABLE_SIZE> gs_history_table_short{};
  std::array<champsim::msl::fwcounter<COUNTER_BITS>, GS_HISTORY_TABLE_SIZE> gs_history_table_medium{};
  std::array<champsim::msl::fwcounter<COUNTER_BITS>, GS_HISTORY_TABLE_SIZE> gs_history_table_long{};

  int weight_short = 1;
  int weight_medium = 1;
  int weight_long = 1;

  using branch_predictor::branch_predictor;

  template<size_t HIST_LEN>
  static std::size_t gs_table_hash(champsim::address ip, std::bitset<HIST_LEN> bh_vector);

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

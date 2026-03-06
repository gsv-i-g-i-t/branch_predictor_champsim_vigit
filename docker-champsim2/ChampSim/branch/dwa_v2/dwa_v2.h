#ifndef DWA_V2_H
#define DWA_V2_H

#include <array>
#include <bitset>

#include "modules.h"
#include "msl/fwcounter.h"

struct dwa_v2 : champsim::modules::branch_predictor {

  static constexpr std::size_t GHIST_SHORT  = 6;
  static constexpr std::size_t GHIST_LONG   = 20;

  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t COUNTER_BITS = 2;

  std::bitset<GHIST_SHORT> hist_short;
  std::bitset<GHIST_LONG>  hist_long;

  std::array<champsim::msl::fwcounter<COUNTER_BITS>, TABLE_SIZE> table_short;
  std::array<champsim::msl::fwcounter<COUNTER_BITS>, TABLE_SIZE> table_long;

  std::array<champsim::msl::fwcounter<2>, TABLE_SIZE> meta_table;

  using branch_predictor::branch_predictor;

  template<size_t HIST_LEN>
  std::size_t hash(champsim::address ip, std::bitset<HIST_LEN> h);

  std::size_t meta_hash(champsim::address ip);

  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address branch_target,
                          bool taken,
                          uint8_t branch_type);
};

#endif

#ifndef IDPN_V3_BP
#define IDPN_V3_BP

#include <array>
#include <bitset>
#include <cstdint>
#include <cmath>
#include "modules.h"

struct idpn_v3 : champsim::modules::branch_predictor {

  static constexpr int GHIST = 512;
  static constexpr int PATH_HIST = 32;

  static constexpr int NUM_TABLES = 4;
  static constexpr int TABLE_SIZE = 4096;

  static constexpr int MAX_W = 127;
  static constexpr int MIN_W = -128;
  static constexpr int THRESH = 64;

  std::bitset<GHIST> ghistory;
  std::array<uint32_t, PATH_HIST> path_history{};

  struct Table {
    std::array<int, TABLE_SIZE> weights{};
  };

  std::array<Table, NUM_TABLES> tables;

  std::array<int, TABLE_SIZE> bias_table{};
  std::array<int, TABLE_SIZE> corrector_table{};

  using branch_predictor::branch_predictor;

  int clamp(int w);
  std::size_t hash(champsim::address ip, int hist_len, int table_id);
  int compute_sum(champsim::address ip);
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address target,
                          bool taken,
                          uint8_t branch_type);
};

#endif

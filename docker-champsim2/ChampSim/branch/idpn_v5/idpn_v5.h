#ifndef IDPN_V5
#define IDPN_V5

#include <array>
#include <bitset>
#include <cmath>
#include "modules.h"

struct idpn_v5 : champsim::modules::branch_predictor {

  static constexpr int GHIST = 512;
  static constexpr int TABLE_SIZE = 4096;
  static constexpr int MAX_W = 7;
  static constexpr int MIN_W = -8;

  // Segment lengths
  static constexpr int L0 = 16;
  static constexpr int L1 = 48;
  static constexpr int L2 = 128;
  static constexpr int L3 = 320;

  std::bitset<GHIST> ghistory;

  struct Table {
    std::array<std::array<int8_t, GHIST>, TABLE_SIZE> weights{};
    std::array<int8_t, TABLE_SIZE> bias{};
  };

  std::array<Table, 4> tables;

  // Adaptive fusion weights
  std::array<int, 4> fusion_weight{4,4,4,4};

  using branch_predictor::branch_predictor;

  std::size_t index(champsim::address ip, int table_id);
  int compute_segment_sum(champsim::address ip, int table_id);
  int compute_total_sum(champsim::address ip);
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address,
                          bool taken,
                          uint8_t);

  inline int clamp(int x);
};

#endif

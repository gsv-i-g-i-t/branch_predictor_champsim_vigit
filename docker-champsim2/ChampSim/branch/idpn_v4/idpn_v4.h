#ifndef IDPN_V4_BP
#define IDPN_V4_BP

#include <array>
#include <bitset>
#include <cstdint>
#include <cmath>
#include "modules.h"

struct idpn_v4 : champsim::modules::branch_predictor {

  static constexpr int GHIST = 256;
  static constexpr int TABLE_SIZE = 4096;
  static constexpr int NUM_TABLES = 4;

  static constexpr int MAX_W = 63;
  static constexpr int MIN_W = -64;
  static constexpr int THRESH = 32;

  std::bitset<GHIST> ghistory;

  uint64_t folded_hist = 0;

  struct Table {
    std::array<int8_t, TABLE_SIZE> w{};
  };

  std::array<Table, NUM_TABLES> tables;

  std::array<int8_t, TABLE_SIZE> bias{};
  std::array<int8_t, TABLE_SIZE> corrector{};

  using branch_predictor::branch_predictor;

  inline int clamp(int x);
  inline std::size_t idx(champsim::address ip, int table_id);
  int compute_sum(champsim::address ip);
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address,
                          bool taken,
                          uint8_t);
};

#endif

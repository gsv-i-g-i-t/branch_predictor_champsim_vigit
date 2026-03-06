#ifndef ATTENTION_BP
#define ATTENTION_BP

#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include "modules.h"

struct attention_bp : champsim::modules::branch_predictor {

  static constexpr int GHIST = 512;
  static constexpr int EMBED_DIM = 16;
  static constexpr int IP_TABLE = 4096;

  std::bitset<GHIST> ghistory;

  // IP embeddings
  std::array<std::array<float, EMBED_DIM>, IP_TABLE> ip_embed{};

  // Projection matrices
  std::array<std::array<float, EMBED_DIM>, EMBED_DIM> Wq{};
  std::array<std::array<float, EMBED_DIM>, EMBED_DIM> Wk{};
  std::array<std::array<float, EMBED_DIM>, EMBED_DIM> Wv{};

  std::array<float, EMBED_DIM> Wo{};

  using branch_predictor::branch_predictor;

  std::size_t ip_index(champsim::address ip);
  float predict_score(champsim::address ip);
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip,
                          champsim::address,
                          bool taken,
                          uint8_t);
};

#endif

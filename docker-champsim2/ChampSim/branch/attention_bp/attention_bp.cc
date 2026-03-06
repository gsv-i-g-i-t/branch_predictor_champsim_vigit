#include "attention_bp.h"

std::size_t attention_bp::ip_index(champsim::address ip) {
  return ip.to<uint64_t>() & (IP_TABLE - 1);
}

float attention_bp::predict_score(champsim::address ip)
{
  auto idx = ip_index(ip);
  auto& embed = ip_embed[idx];

  // Compute Query
  float Q[EMBED_DIM] = {0};
  for (int i = 0; i < EMBED_DIM; i++)
    for (int j = 0; j < EMBED_DIM; j++)
      Q[i] += Wq[i][j] * embed[j];

  float context[EMBED_DIM] = {0};

  // Attention over history
  for (int h = 0; h < GHIST; h++) {

    float hist_val = ghistory[h] ? 1.0f : -1.0f;

    float K[EMBED_DIM] = {0};
    float V[EMBED_DIM] = {0};

    for (int i = 0; i < EMBED_DIM; i++) {
      K[i] = Wk[i][0] * hist_val;
      V[i] = Wv[i][0] * hist_val;
    }

    float score = 0;
    for (int i = 0; i < EMBED_DIM; i++)
      score += Q[i] * K[i];

    float weight = std::tanh(score);  // approximate softmax

    for (int i = 0; i < EMBED_DIM; i++)
      context[i] += weight * V[i];
  }

  float out = 0;
  for (int i = 0; i < EMBED_DIM; i++)
    out += Wo[i] * context[i];

  return out;
}

bool attention_bp::predict_branch(champsim::address ip)
{
  return predict_score(ip) >= 0;
}

void attention_bp::last_branch_result(champsim::address ip,
                                      champsim::address,
                                      bool taken,
                                      uint8_t)
{
  float lr = 0.0005f;

  float score = predict_score(ip);
  float target = taken ? 1.0f : -1.0f;
  float error = target - score;

  auto idx = ip_index(ip);

  // Update output layer
  for (int i = 0; i < EMBED_DIM; i++)
    Wo[i] += lr * error;

  // Update IP embedding (very simplified SGD)
  for (int i = 0; i < EMBED_DIM; i++)
    ip_embed[idx][i] += lr * error;

  ghistory <<= 1;
  ghistory[0] = taken;
}

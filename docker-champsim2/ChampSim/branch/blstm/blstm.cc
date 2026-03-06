#include "blstm.h"

std::size_t blstm::ip_index(champsim::address ip) {
  return ip.to<uint64_t>() & (IP_TABLE - 1);
}

float blstm::sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float blstm::tanh_clip(float x) {
  return std::tanh(x);
}

float blstm::predict_score(champsim::address ip)
{
  auto idx = ip_index(ip);
  auto& h_state = h[idx];

  float out = 0;
  for (int i = 0; i < HIDDEN; i++)
    out += Wout[i] * h_state[i];

  return out;
}

bool blstm::predict_branch(champsim::address ip)
{
  return predict_score(ip) >= 0;
}

void blstm::last_branch_result(champsim::address ip,
                               champsim::address,
                               bool taken,
                               uint8_t)
{
  float lr = 0.001f;
  float input = taken ? 1.0f : -1.0f;
  float target = input;

  auto idx = ip_index(ip);
  auto& h_state = h[idx];
  auto& c_state = c[idx];

  float new_h[HIDDEN];
  float new_c[HIDDEN];

  // LSTM forward
  for (int i = 0; i < HIDDEN; i++) {

    float i_gate = bi[i];
    float f_gate = bf[i];
    float o_gate = bo[i];
    float g_gate = bg[i];

    for (int j = 0; j < HIDDEN; j++) {
      i_gate += Wi[i][j] * h_state[j];
      f_gate += Wf[i][j] * h_state[j];
      o_gate += Wo_gate[i][j] * h_state[j];
      g_gate += Wg[i][j] * h_state[j];
    }

    i_gate = sigmoid(i_gate);
    f_gate = sigmoid(f_gate);
    o_gate = sigmoid(o_gate);
    g_gate = tanh_clip(g_gate);

    new_c[i] = f_gate * c_state[i] + i_gate * g_gate;
    new_h[i] = o_gate * tanh_clip(new_c[i]);
  }

  float score = 0;
  for (int i = 0; i < HIDDEN; i++)
    score += Wout[i] * new_h[i];

  float error = target - score;

  // Update output layer
  for (int i = 0; i < HIDDEN; i++)
    Wout[i] += lr * error * new_h[i];

  // Simple gradient on hidden (very truncated BPTT)
  for (int i = 0; i < HIDDEN; i++)
    new_h[i] += lr * error;

  // Commit state
  for (int i = 0; i < HIDDEN; i++) {
    h_state[i] = new_h[i];
    c_state[i] = new_c[i];
  }
}

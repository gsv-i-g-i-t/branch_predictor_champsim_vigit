#include "dcn.h"

int dcn::clamp(int w) {
  if (w > MAX_WEIGHT) return MAX_WEIGHT;
  if (w < MIN_WEIGHT) return MIN_WEIGHT;
  return w;
}

bool dcn::predict_branch(champsim::address)
{
  std::array<int, NUM_LAYERS> layer_outputs;

  for (int l = 0; l < NUM_LAYERS; l++) {
    int sum = biases[l];

    for (int k = 0; k < KERNEL_SIZE; k++) {
      int idx = k * dilations[l];
      if (idx >= HISTORY_LEN) break;

      int bit = ghistory[idx] ? 1 : -1;
      sum += weights[l][k] * bit;
    }

    // ReLU
    if (sum < 0) sum = 0;

    layer_outputs[l] = sum;
  }

  int final_sum = final_bias;

  for (int l = 0; l < NUM_LAYERS; l++)
    final_sum += final_weights[l] * layer_outputs[l];

  return final_sum >= 0;
}

void dcn::last_branch_result(champsim::address ip, champsim::address, bool taken, uint8_t)
{
  bool prediction = predict_branch(ip);
  int target = taken ? 1 : -1;

  std::array<int, NUM_LAYERS> layer_outputs;

  for (int l = 0; l < NUM_LAYERS; l++) {
    int sum = biases[l];
    for (int k = 0; k < KERNEL_SIZE; k++) {
      int idx = k * dilations[l];
      if (idx >= HISTORY_LEN) break;
      int bit = ghistory[idx] ? 1 : -1;
      sum += weights[l][k] * bit;
    }
    if (sum < 0) sum = 0;
    layer_outputs[l] = sum;
  }

  int final_sum = final_bias;
  for (int l = 0; l < NUM_LAYERS; l++)
    final_sum += final_weights[l] * layer_outputs[l];

  int confidence = abs(final_sum);

  if (prediction != taken || confidence < THRESHOLD) {

    // Update final layer
    final_bias = clamp(final_bias + target);

    for (int l = 0; l < NUM_LAYERS; l++)
      final_weights[l] = clamp(final_weights[l] + target * layer_outputs[l]);

    // Update convolution layers
    for (int l = 0; l < NUM_LAYERS; l++) {
      biases[l] = clamp(biases[l] + target);

      for (int k = 0; k < KERNEL_SIZE; k++) {
        int idx = k * dilations[l];
        if (idx >= HISTORY_LEN) break;

        int bit = ghistory[idx] ? 1 : -1;
        weights[l][k] = clamp(weights[l][k] + target * bit);
      }
    }
  }

  ghistory <<= 1;
  ghistory[0] = taken;
}

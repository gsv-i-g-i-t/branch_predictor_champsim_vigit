#include "dwa.h"

template<size_t HIST_LEN>
std::size_t dwa::gs_table_hash(champsim::address ip,
                          std::bitset<HIST_LEN> bh_vector)
{
    std::size_t hash = bh_vector.to_ullong();
    hash ^= static_cast<std::size_t>(ip.to<uint64_t>());
    return hash % dwa::GS_HISTORY_TABLE_SIZE;
}

bool dwa::predict_branch(champsim::address ip)
{
  auto gs_hash_short  = gs_table_hash(ip, branch_history_vector_short);
  auto gs_hash_medium = gs_table_hash(ip, branch_history_vector_medium);
  auto gs_hash_long   = gs_table_hash(ip, branch_history_vector_long);

  auto& value_short  = gs_history_table_short[gs_hash_short];
  auto& value_medium = gs_history_table_medium[gs_hash_medium];
  auto& value_long   = gs_history_table_long[gs_hash_long];

  bool pred_short  = value_short.value()  >= value_short.maximum / 2;
  bool pred_medium = value_medium.value() >= value_medium.maximum / 2;
  bool pred_long   = value_long.value()   >= value_long.maximum / 2;

  int score = weight_short*pred_short +
              weight_medium*pred_medium +
              weight_long*pred_long;

  return score >= (weight_short + weight_medium + weight_long) / 2;
}

static int clamp(int w)
{
  if (w < 0) return 0;
  if (w > 7) return 7;
  return w;
}

void dwa::last_branch_result(champsim::address ip, champsim::address, bool taken, uint8_t)
{
  auto gs_hash_short  = gs_table_hash(ip, branch_history_vector_short);
  auto gs_hash_medium = gs_table_hash(ip, branch_history_vector_medium);
  auto gs_hash_long   = gs_table_hash(ip, branch_history_vector_long);

  gs_history_table_short[gs_hash_short]  += taken ? 1 : -1;
  gs_history_table_medium[gs_hash_medium] += taken ? 1 : -1;
  gs_history_table_long[gs_hash_long]    += taken ? 1 : -1;

  auto& value_short  = gs_history_table_short[gs_hash_short];
  auto& value_medium = gs_history_table_medium[gs_hash_medium];
  auto& value_long   = gs_history_table_long[gs_hash_long];

  bool pred_short  = value_short.value()  >= value_short.maximum / 2;
  bool pred_medium = value_medium.value() >= value_medium.maximum / 2;
  bool pred_long   = value_long.value()   >= value_long.maximum / 2;

  weight_short  = clamp(weight_short  + (pred_short  == taken ? 1 : -1));
  weight_medium = clamp(weight_medium + (pred_medium == taken ? 1 : -1));
  weight_long   = clamp(weight_long   + (pred_long   == taken ? 1 : -1));

  branch_history_vector_short <<= 1;
  branch_history_vector_medium <<= 1;
  branch_history_vector_long <<= 1;

  branch_history_vector_short[0]  = taken;
  branch_history_vector_medium[0] = taken;
  branch_history_vector_long[0]   = taken;
}

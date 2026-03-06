#include "dwa_v2.h"

template<size_t HIST_LEN>
std::size_t dwa_v2::hash(champsim::address ip, std::bitset<HIST_LEN> h)
{
    std::size_t ip_val = ip.to<std::size_t>();
    std::size_t hist_val = h.to_ullong();
    return (ip_val ^ hist_val) % TABLE_SIZE;
}

std::size_t dwa_v2::meta_hash(champsim::address ip)
{
    return ip.to<std::size_t>() % TABLE_SIZE;
}

bool dwa_v2::predict_branch(champsim::address ip)
{
    auto idx_s = hash(ip, hist_short);
    auto idx_l = hash(ip, hist_long);
    auto idx_meta = meta_hash(ip);

    bool p_s = table_short[idx_s].value() >= table_short[idx_s].maximum/2;
    bool p_l = table_long[idx_l].value()  >= table_long[idx_l].maximum/2;

    int meta = meta_table[idx_meta].value();

    return (meta <= 1) ? p_s : p_l;
}

void dwa_v2::last_branch_result(champsim::address ip,
                                champsim::address branch_target,
                                bool taken,
                                uint8_t branch_type)
{
    auto idx_s = hash(ip, hist_short);
    auto idx_l = hash(ip, hist_long);
    auto idx_meta = meta_hash(ip);

    bool p_s = table_short[idx_s].value() >= table_short[idx_s].maximum/2;
    bool p_l = table_long[idx_l].value()  >= table_long[idx_l].maximum/2;

    table_short[idx_s] += taken ? 1 : -1;
    table_long[idx_l]  += taken ? 1 : -1;

    if (p_s != p_l) {
        if (p_s == taken)
            meta_table[idx_meta] -= 1;
        else
            meta_table[idx_meta] += 1;
    }

    hist_short <<= 1; hist_short[0] = taken;
    hist_long  <<= 1; hist_long[0]  = taken;
}

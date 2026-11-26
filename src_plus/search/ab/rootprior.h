#pragma once

#include "../../core/pos.h"
#include "../../core/types.h"

#include <array>
#include <string_view>

namespace Search {
class SearchThread;
}

namespace Search::AB {

struct RootPrior
{
    Value value = VALUE_NONE;
    bool orderingOnly = false;
    std::array<Score, FULL_BOARD_CELL_COUNT> policy;
};

void prepareRootPrior(SearchThread &thread, RootPrior &prior, std::string_view experiment);
Pos prepareRootProof(SearchThread &thread, Value &mate);

}  // namespace Search::AB

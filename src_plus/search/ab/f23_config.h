// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// All playing objects share one ABSearchData layout. Only search.cpp and
// history.cpp use these switches, so a cloud ablation may relink those two units.
#ifndef RAPFI_F23_MASK
#define RAPFI_F23_MASK 0
#endif

#define F23_F15        (RAPFI_F23_MASK & 1)
#define F23_SINGULAR   (RAPFI_F23_MASK & 2)
#define F23_MALUS      (RAPFI_F23_MASK & 4)
#define F23_CORRECTION (RAPFI_F23_MASK & 8)
#define F23_STABILITY  (RAPFI_F23_MASK & 16)

// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright The XCSoar Project

#pragma once

#include "ui/canvas/Color.hpp"

static constexpr Color COLOR_XCSOAR_LIGHT = Color(0xf0, 0xc8, 0x90);
static constexpr Color COLOR_XCSOAR = Color(0xc8, 0x73, 0x2a);
static constexpr Color COLOR_XCSOAR_DARK = Color(0x6f, 0x3a, 0x12);

/**
 * Dark mode color palette derived from the XCSoar brand color.
 */
static constexpr Color COLOR_DARK_THEME_BACKGROUND =
  Color(0x1f, 0x16, 0x12);
static constexpr Color COLOR_DARK_THEME_CAPTION =
  Color(0x2c, 0x1f, 0x18);
static constexpr Color COLOR_DARK_THEME_CAPTION_INACTIVE =
  Color(0x4a, 0x38, 0x30);
static constexpr Color COLOR_DARK_THEME_LIST =
  Color(0x3a, 0x2b, 0x23);
static constexpr Color COLOR_DARK_THEME_LIST_SELECTED =
  Color(0x54, 0x3f, 0x33);
static constexpr Color COLOR_DARK_THEME_BUTTON =
  Color(0x5f, 0x3d, 0x28);
static constexpr Color COLOR_DARK_THEME_GRADIENT_TOP =
  Color(0x3a, 0x27, 0x1d);

/**
 * Light mode dialog background colors (warm parchment tint).
 */
static constexpr Color COLOR_DIALOG_BACKGROUND =
  Color(0xe2, 0xdc, 0xbe);
static constexpr Color COLOR_DIALOG_GRADIENT_TOP =
  Color(0xf0, 0xeb, 0xd4);

/**
 * Admonition colors for Markdown rendering.
 */
static constexpr Color COLOR_ADMONITION_IMPORTANT =
  Color(0xd0, 0x6b, 0x00);
static constexpr Color COLOR_ADMONITION_IMPORTANT_DARK =
  Color(0xff, 0xa0, 0x30);
static constexpr Color COLOR_ADMONITION_TIP =
  Color(0x00, 0x80, 0x00);

/**
 * A muted green readable on light backgrounds.
 * Standard COLOR_GREEN (0,255,0) is too bright on white.
 */
static constexpr Color COLOR_LIGHT_GREEN = Color(0x00, 0xc0, 0x00);

static constexpr uint8_t ALPHA_OVERLAY = 0xA0;

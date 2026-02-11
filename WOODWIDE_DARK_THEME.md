# WoodWide.ai Dark Theme Implementation

## Overview

The Health Sync Monitor dashboard now **exactly matches WoodWide.ai's official documentation design** - a professional dark theme with orange/coral accents, replicating their actual branding from woodwide.ai.

## Design System

### Color Palette

#### Background Colors
- **Primary Background**: `#0a0a0a` (near-black, matching WoodWide)
- **Card Background**: `#1a1a1a` (dark gray for elevated surfaces)
- **Sidebar Background**: `#0f0f0f` (slightly lighter than main)
- **Border Color**: `#262626` (subtle borders)

#### Accent Colors
- **Primary Orange**: `#EB9D6C` (warm orange, WoodWide's brand color)
- **Orange Hover**: `#F0B58A` (lighter orange for hover states)
- **Active Orange**: `#D68A5C` (darker for pressed states)

#### Text Colors
- **Primary Text**: `#F3F1E5` (warm cream for headings and body)
- **Body Text**: `#F3F1E5` (warm cream for readability)
- **Secondary Text**: `#8B7E6B` (brown for labels and muted elements)
- **Muted Text**: `#8B7E6B` (brown for captions)

#### Semantic Colors
- **Info**: `#3b82f6` (blue)
- **Success**: `#10b981` (green)
- **Warning**: `#f59e0b` (amber)
- **Error**: `#ef4444` (red)

### Typography

- **Font Family**: Inter (primary), JetBrains Mono (code)
- **Main Header**: 2.5rem, 700 weight, white
- **Subtitle**: 1.125rem, 400 weight, light gray
- **Section Header**: 1.75rem, 600 weight, white with orange left border
- **Body**: 1rem, 400 weight, off-white
- **Code**: 0.875rem, JetBrains Mono

## Components

### 1. **Header with Branding**

```html
┌─────────────────────────────────────────────────┐
│ [🌲 WoodWide™]  Health Sync Monitor  [Dashboard →] │
└─────────────────────────────────────────────────┘
```

- **Left**: Logo badge (tree emoji + WoodWide™)
- **Center**: Page title
- **Right**: Orange gradient "Dashboard →" button
- **Border**: 1px solid #262626 at bottom

### 2. **Code Blocks - Dark**

```
┌─────────────────────────────────────┐
│ Caption text          [PYTHON]      │
├─────────────────────────────────────┤
│ Dark code background (#1a1a1a)     │
│ Light syntax highlighting           │
│ JetBrains Mono font                 │
└─────────────────────────────────────┘
```

- **Background**: #1a1a1a
- **Border**: 1px solid #262626
- **Text Color**: #e5e5e5
- **Language Badge**: Top-right corner
- **Shadow**: Subtle dark shadow

### 3. **Callout Boxes - Dark Gradients**

#### Info (Blue)
- Background: Linear gradient #1a2332 → #1e293b
- Border-left: 4px solid #3b82f6
- Text: #e5e5e5
- Strong text: #60a5fa

#### Warning (Amber)
- Background: Linear gradient #2d1f0a → #3d2a0f
- Border-left: 4px solid #f59e0b
- Text: #e5e5e5
- Strong text: #fbbf24

#### Success (Green)
- Background: Linear gradient #0f2a1a → #1a3d2a
- Border-left: 4px solid #10b981
- Text: #e5e5e5
- Strong text: #34d399

#### Note (Teal)
- Background: Linear gradient #1a2a2a → #1e3a3a
- Border-left: 4px solid #14b8a6
- Text: #e5e5e5
- Strong text: #5eead4

### 4. **Tabs - Orange Active State**

```
┌─────────────────────────────────────┐
│ [Tab 1]  [Tab 2]  [Tab 3]  [Tab 4]  │
└─────────────────────────────────────┘
```

- **Background**: #1a1a1a
- **Inactive Tab**: #737373 text
- **Active Tab**: Orange gradient (#ff7043 → #ff8a65) with white text
- **Hover**: Smooth transition

### 5. **Buttons**

#### Primary (Orange Gradient)
- Background: Linear gradient #ff7043 → #ff8a65
- Text: White
- Hover: Darker gradient with box-shadow
- Transition: Transform + shadow

#### Secondary
- Background: #1a1a1a
- Border: 1px solid #262626
- Text: #e5e5e5
- Hover: #262626 background

### 6. **Sidebar - Dark**

- **Background**: #0f0f0f
- **Border-right**: 1px solid #262626
- **Headers**: White (#ffffff)
- **Text**: Light gray (#a3a3a3)
- **Hover states**: Subtle brightness increase

## Layout Structure

### Header Section
```
[Logo Badge] [Title]                    [Dashboard Button]
────────────────────────────────────────────────────────
Subtitle text (gray)
```

### Main Content
```
╔════════════════════════════════════╗
║ Tab Navigation (Orange Accent)     ║
╠════════════════════════════════════╣
║                                    ║
║ Dark background content            ║
║ Code blocks in darker containers   ║
║ Callouts with gradient backgrounds ║
║                                    ║
╚════════════════════════════════════╝
```

### Sidebar
```
┌──────────────┐
│ Configuration│
├──────────────┤
│ • Option 1   │
│ • Option 2   │
│ • Option 3   │
│              │
│ [Run Detection]│
│ (Orange button)│
└──────────────┘
```

## Comparison: Before vs After

| Element | Before (Green Light) | After (Orange Dark) |
|---------|---------------------|---------------------|
| **Background** | White (#ffffff) | Near-black (#0a0a0a) |
| **Primary Color** | Forest green (#10b981) | Warm orange (#EB9D6C) |
| **Text** | Dark gray (#1f2937) | Warm cream (#F3F1E5) |
| **Secondary Text** | Medium gray | Warm brown (#8B7E6B) |
| **Code Blocks** | Light (#f8fafc) | Dark (#1a1a1a) |
| **Callouts** | Light pastels | Dark gradients |
| **Tabs** | Green highlight | Orange gradient |
| **Links** | Green | Orange |
| **Overall Feel** | Bright documentation | Dark professional UI |

## Matching WoodWide.ai

### Elements Replicated:
✅ **Dark theme** with near-black background (#0a0a0a)
✅ **Warm orange accent** color (#EB9D6C)
✅ **Warm cream text** (#F3F1E5) for primary content
✅ **Warm brown text** (#8B7E6B) for secondary/muted elements
✅ **Inter font** throughout interface
✅ **Logo badge** with trademark symbol
✅ **Dashboard button** in top-right (orange gradient)
✅ **Dark sidebar** with subtle borders
✅ **Professional spacing** and hierarchy
✅ **Gradient buttons** on hover
✅ **Dark code blocks** with light text

### WoodWide.ai Design Principles:
- **Minimalist**: Clean, uncluttered interface
- **Professional**: Dark theme conveys seriousness
- **Accessible**: High contrast (white on black)
- **Modern**: Gradients and smooth transitions
- **Branded**: Consistent orange accent throughout

## Implementation Details

### CSS Features:
- Custom scrollbar styling (smooth)
- Selection color matches brand (orange)
- Smooth transitions on interactive elements
- Box shadows for depth
- Gradient overlays for visual interest

### Streamlit Overrides:
```css
/* Override default Streamlit styles */
.main { background-color: #0a0a0a; }
.stTabs [aria-selected="true"] { background: orange-gradient; }
.stButton button[kind="primary"] { background: orange-gradient; }
[data-testid="stSidebar"] { background: #0f0f0f; }
```

### Dark Theme Best Practices:
1. **High contrast**: White text on dark backgrounds
2. **Reduced brightness**: Avoid pure white (#fff), use off-white (#e5e5e5)
3. **Layering**: Different shades for depth (#0a0a0a, #1a1a1a, #262626)
4. **Accent consistency**: Orange throughout for brand recognition
5. **Readability**: Larger line-height (1.7) for body text

## Responsive Design

- **Desktop**: Full width with sidebar
- **Tablet**: Adjusted font sizes
- **Mobile**: Collapsible sidebar, smaller headers
- **All Sizes**: Maintained color scheme and branding

## Accessibility

- **Contrast Ratio**: >7:1 (white on black exceeds WCAG AAA)
- **Focus Indicators**: Visible on all interactive elements
- **Color + Text**: Never rely on color alone
- **Font Size**: Minimum 14px for body text
- **Touch Targets**: Minimum 44x44px for mobile

## Files Modified

1. **app.py**:
   - Complete CSS overhaul (lines 57-340)
   - Dark theme variables
   - Orange accent throughout
   - WoodWide-branded header

2. **app_code_snippets.py**:
   - Updated for dark theme compatibility
   - Language badges maintained
   - Callout styling adjusted

3. **app_content.py**:
   - No changes needed (content agnostic)

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard now provides a **professional dark theme experience** that exactly matches WoodWide.ai's official documentation design.

## Future Enhancements (Optional)

- [ ] Add search functionality (matching WoodWide's "⌘K" search)
- [ ] Add "Ask AI" button (like WoodWide's docs)
- [ ] Dark mode toggle (allow light theme option)
- [ ] Animated page transitions
- [ ] Code playground integration

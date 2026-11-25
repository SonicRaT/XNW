# Third-party notices

XNW is licensed under the GNU General Public License, version 3 or any later version (see [LICENSE](LICENSE)).
Parts of it derive from other open-source work. Each ported file names its origin in its header; the notices
those licences ask for are collected here.

## FFmpeg

`crates/xnw-formats/src/xma.rs` and `crates/xnw-formats/src/xma/tables.rs` derive from FFmpeg's
`libavcodec/wmaprodec.c` and `libavcodec/wmaprodata.h`.

```
Wmapro compatible decoder
Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
Copyright (c) 2008 - 2011 Sascha Sommer, Benjamin Larsson

WMA 9/3/PRO compatible decoder
Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
Copyright (c) 2008 - 2009 Sascha Sommer
```

FFmpeg is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General
Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version. As that licence permits, the derived code is carried here under the GNU General
Public License, version 3 or any later version.

## Xbox2Windows

A reference test vector in `crates/xnw-formats/src/bcpack.rs` comes from Xbox2Windows.

```
MIT License

Copyright (c) 2026 Xbox2Windows contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## VIctory

Much of the runtime design and parts of the code are ported from VIctory, licensed under the GNU General Public
License, version 3 or (at your option) any later version, the same licence as XNW. VIctory states its licence as
"GPL-3.0-or-later" (its README and workspace manifest) and names no copyright holder. Most ports are from commit
`d2a4bae`; every ported file names the project, the upstream file and the commit it was taken from in its header.

## AMD address library (addrlib, as shipped in Mesa)

`crates/xnw-formats/src/texture.rs` follows the algorithms of AMD's address library (`src/amd/addrlib` in Mesa),
by way of VIctory.

```
Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

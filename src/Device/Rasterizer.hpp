// Copyright 2016 The SwiftShader Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef sw_Rasterizer_hpp
#define sw_Rasterizer_hpp

#include "PixelProcessor.hpp"
#include "Device/Config.hpp"

namespace sw {

class Rasterizer : public RasterizerFunction
{
public:
	Rasterizer()
	    : device(Arg<0>())
	    , primitive(Arg<1>())
	    , count(Arg<2>())
	    , data(Arg<3>())
#if USE_SCANLINE_RASTERIZER
	    , cluster(Arg<4>())
	    , clusterCount(Arg<5>())
#else
	    , tile(Arg<4>())
	    , tileQueue(Arg<5>())
#endif
	{}
	virtual ~Rasterizer() {}

protected:
	Pointer<Byte> device;
	Pointer<Byte> primitive;
	Int count;
	Pointer<Byte> data;
#if USE_SCANLINE_RASTERIZER
	Int cluster;
	Int clusterCount;
#else
	Pointer<Byte> tile;
	Pointer<Byte> tileQueue;
#endif
};

}  // namespace sw

#endif  // sw_Rasterizer_hpp

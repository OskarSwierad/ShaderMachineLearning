import sgl.device.print;

struct GlobalParams {
    uint2 Dimensions;
    RWTexture2D<float4> OutTexture;
};
ParameterBlock<GlobalParams> Params;

[shader("compute")]
[numthreads(16, 16, 1)]
void MainCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (all(dispatchThreadID == uint3(0, 0, 0)))
    {
        print("--- Out dim: {} px", Params.Dimensions);
        uint2 outDims;
        Params.OutTexture.GetDimensions(outDims.x, outDims.y);
        print("--- Out tex: {} x {} px", outDims.x, outDims.y);
    }
    
    const uint2 PixelCoord = dispatchThreadID.xy;
    if (any(PixelCoord >= Params.Dimensions))
    {
        return;
    }

    const float2 UV = frac(float2(PixelCoord) / float2(Params.Dimensions));
    Params.OutTexture[PixelCoord] = float4(UV, 0, 1);
}

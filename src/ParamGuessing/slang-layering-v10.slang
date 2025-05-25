// Blends 5 image layers
// Each of them has a mask - a linear gradient, with a user-specified range and rotation

#define DIM_X 512
#define DIM_Y 512
#define DegreesToRadians (3.14159265358979323846 / 180.0)

struct GlobalParams
{
    uint2 ScreenSize;

    float3  Layer0_TintColor;
    float   Layer0_TilingScale;
    float2  Layer0_TilingOffset;

    float3  Layer1_TintColor;
    float   Layer1_TilingScale;
    float2  Layer1_TilingOffset;
    float   Layer1_MaskGradientStart;
    float   Layer1_MaskGradientEnd;
    float   Layer1_MaskGradientAngle;
    float   Layer1_LayerOpacity;

    float3  Layer2_TintColor;
    float   Layer2_TilingScale;
    float2  Layer2_TilingOffset;
    float   Layer2_MaskGradientStart;
    float   Layer2_MaskGradientEnd;
    float   Layer2_MaskGradientAngle;
    float   Layer2_LayerOpacity;

    float3  Layer3_TintColor;
    float   Layer3_TilingScale;
    float2  Layer3_TilingOffset;
    float   Layer3_MaskGradientStart;
    float   Layer3_MaskGradientEnd;
    float   Layer3_MaskGradientAngle;
    float   Layer3_LayerOpacity;

    float3  Layer4_TintColor;
    float   Layer4_TilingScale;
    float2  Layer4_TilingOffset;
    float   Layer4_MaskGradientStart;
    float   Layer4_MaskGradientEnd;
    float   Layer4_MaskGradientAngle;
    float   Layer4_LayerOpacity;
};
ConstantBuffer<GlobalParams> GlobalParamsCB : register(b0);

Texture2D<float4> Texture0 : register(t0);
Texture2D<float4> Texture1 : register(t1);
Texture2D<float4> Texture2 : register(t2);
Texture2D<float4> Texture3 : register(t3);
Texture2D<float4> Texture4 : register(t4);

SamplerState SamplerWrapped : register(s0);

RWTexture2D<float4> OutputTexture : register(u0);

float CalculateGradientMask(
    const float2 UvCoordinates,
    const float  MaskGradientStart,
    const float  MaskGradientEnd,
    const float  MaskGradientAngle)
{
    const float AngleRadians = MaskGradientAngle * (float)DegreesToRadians;
    const float2 DirectionVector = normalize(float2(cos(AngleRadians), sin(AngleRadians)));

    const float MinCornerProjection = min(0.0f, DirectionVector.x) + min(0.0f, DirectionVector.y);
    const float MaxCornerProjection = max(0.0f, DirectionVector.x) + max(0.0f, DirectionVector.y);
    const float ScreenProjectedRange = MaxCornerProjection - MinCornerProjection;

    const float ProjectedUvOnDirection = dot(UvCoordinates, DirectionVector);

    const float NormalizedScreenProjection = (abs(ScreenProjectedRange) > 1e-6f)
                                            ? (ProjectedUvOnDirection - MinCornerProjection) / ScreenProjectedRange
                                            : 1.0f; // Match Slang's handling of range ~= 0

    const float GradientRange = MaskGradientEnd - MaskGradientStart;
    float GradientMaskValue;
    if (abs(GradientRange) > 0.0)
    {
        GradientMaskValue = (NormalizedScreenProjection - MaskGradientStart) / GradientRange;
        GradientMaskValue = saturate(GradientMaskValue);
    }
    else
    {
        // Sharp cutoff if gradient start and end are the same
        GradientMaskValue = float(NormalizedScreenProjection >= MaskGradientStart);
    }
    return GradientMaskValue;
}


void ProcessLayerFunc(
    Texture2D<float4> LayerTexture,
    const float3 TintColor,
    const float TilingScale,
    const float2 TilingOffset,
    const float MaskGradientStart,
    const float MaskGradientEnd,
    const float MaskGradientAngle,
    const float LayerOpacity,
    const float2 BaseTexCoord,
    inout float3 FinalColor,
    inout float BudgetLeft)
{
    const float2 TiledUV = BaseTexCoord * TilingScale + TilingOffset;
    const float4 SampledColor = LayerTexture.SampleLevel(SamplerWrapped, TiledUV, 0);

    const float3 BackgroundColor = float3(0.5f, 0.5f, 0.5f);
    const float3 ImageColor = lerp(BackgroundColor, SampledColor.rgb, SampledColor.a);

    const float3 DesaturationWeights = float3(0.3f, 0.5f, 0.2f);
    const float ImageDesaturated = dot(ImageColor, DesaturationWeights);

    const float3 TintedColor = ImageDesaturated * TintColor;

    const float MaskValue = CalculateGradientMask(
        BaseTexCoord, MaskGradientStart, MaskGradientEnd, MaskGradientAngle
    );

    const float LayerAlpha = MaskValue * LayerOpacity;
    const float LayerAlphaFinal = min(BudgetLeft, LayerAlpha);

    FinalColor += TintedColor * LayerAlphaFinal;
    BudgetLeft -= LayerAlphaFinal;
}

#define ProcessLayer(idx) { ProcessLayerFunc(              \
            Texture##idx,                                  \
            GlobalParamsCB.Layer##idx##_TintColor,         \
            GlobalParamsCB.Layer##idx##_TilingScale,       \
            GlobalParamsCB.Layer##idx##_TilingOffset,      \
            GlobalParamsCB.Layer##idx##_MaskGradientStart, \
            GlobalParamsCB.Layer##idx##_MaskGradientEnd,   \
            GlobalParamsCB.Layer##idx##_MaskGradientAngle, \
            GlobalParamsCB.Layer##idx##_LayerOpacity,      \
            TexCoord, FinalColor, BudgetLeft); }

float3 BlendLayers(const uint2 PixelCoordFlipped)
{
    const float2 TexCoord = (float2(PixelCoordFlipped) + 0.5f) / float2(DIM_X, DIM_Y);

    float3 FinalColor = 0.0f;
    float BudgetLeft = 1.0f;
    ProcessLayer(4);
    ProcessLayer(3);
    ProcessLayer(2);
    ProcessLayer(1);
    ProcessLayerFunc( // Layer 0
        Texture0, GlobalParamsCB.Layer0_TintColor,
        GlobalParamsCB.Layer0_TilingScale, GlobalParamsCB.Layer0_TilingOffset,
        0.0f, 0.000001f, 0.0f, 1.0f,
        TexCoord, FinalColor, BudgetLeft
    );

    return saturate(FinalColor);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void MainCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 PixelCoord = dispatchThreadID.xy;
    if (any(PixelCoord.xy >= uint2(DIM_X, DIM_Y)))
    {
        return;
    }

    const uint2 PixelCoordFlipped = uint2(PixelCoord.x, DIM_Y - PixelCoord.y);
    
    const float3 FinalColorOut = BlendLayers(PixelCoordFlipped);

    OutputTexture[PixelCoord] = float4(FinalColorOut, 1.0f);
}

float4 imageMain(uint2 PixelCoord, uint2 ScreenSize)
{
    MainCS(uint3(PixelCoord, 0));
    return float4(OutputTexture[PixelCoord].rgb, 1.0);
}

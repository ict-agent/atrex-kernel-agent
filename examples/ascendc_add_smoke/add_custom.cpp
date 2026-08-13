// Minimal generated AscendC device kernel used by the 910B1 toolchain smoke test.
#include "kernel_operator.h"

namespace AtrexAscendSmoke {

constexpr int32_t kElementCount = 8 * 2048;
constexpr int32_t kBlockCount = 8;
constexpr int32_t kElementsPerBlock = kElementCount / kBlockCount;
constexpr int32_t kTilesPerBlock = 4;
constexpr int32_t kElementsPerTile = kElementsPerBlock / kTilesPerBlock;
constexpr int32_t kQueueDepth = 1;

class VectorAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR output)
    {
        const int32_t blockOffset = AscendC::GetBlockIdx() * kElementsPerBlock;
        xGlobal_.SetGlobalBuffer((__gm__ half *)x + blockOffset, kElementsPerBlock);
        yGlobal_.SetGlobalBuffer((__gm__ half *)y + blockOffset, kElementsPerBlock);
        outputGlobal_.SetGlobalBuffer((__gm__ half *)output + blockOffset, kElementsPerBlock);

        pipe_.InitBuffer(xQueue_, kQueueDepth, kElementsPerTile * sizeof(half));
        pipe_.InitBuffer(yQueue_, kQueueDepth, kElementsPerTile * sizeof(half));
        pipe_.InitBuffer(outputQueue_, kQueueDepth, kElementsPerTile * sizeof(half));
    }

    __aicore__ inline void Run()
    {
        for (int32_t tile = 0; tile < kTilesPerBlock; ++tile) {
            CopyInputs(tile);
            AddTile();
            CopyOutput(tile);
        }
    }

private:
    __aicore__ inline void CopyInputs(int32_t tile)
    {
        auto xLocal = xQueue_.AllocTensor<half>();
        auto yLocal = yQueue_.AllocTensor<half>();
        const int32_t offset = tile * kElementsPerTile;
        AscendC::DataCopy(xLocal, xGlobal_[offset], kElementsPerTile);
        AscendC::DataCopy(yLocal, yGlobal_[offset], kElementsPerTile);
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void AddTile()
    {
        auto xLocal = xQueue_.DeQue<half>();
        auto yLocal = yQueue_.DeQue<half>();
        auto outputLocal = outputQueue_.AllocTensor<half>();
        AscendC::Add(outputLocal, xLocal, yLocal, kElementsPerTile);
        outputQueue_.EnQue(outputLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutput(int32_t tile)
    {
        auto outputLocal = outputQueue_.DeQue<half>();
        AscendC::DataCopy(
            outputGlobal_[tile * kElementsPerTile], outputLocal, kElementsPerTile
        );
        outputQueue_.FreeTensor(outputLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, kQueueDepth> yQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, kQueueDepth> outputQueue_;
    AscendC::GlobalTensor<half> xGlobal_;
    AscendC::GlobalTensor<half> yGlobal_;
    AscendC::GlobalTensor<half> outputGlobal_;
};

}  // namespace AtrexAscendSmoke

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR output)
{
    AtrexAscendSmoke::VectorAdd kernel;
    kernel.Init(x, y, output);
    kernel.Run();
}

#ifndef ASCENDC_CPU_DEBUG
void add_custom_do(
    uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *output
)
{
    add_custom<<<blockDim, nullptr, stream>>>(x, y, output);
}
#endif


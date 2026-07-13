#pragma once

class RenderGraph;

class Pass {
    virtual void AddToGraph(RenderGraph&) = 0;
};
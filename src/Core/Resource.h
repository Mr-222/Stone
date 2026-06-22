#pragma once

class Resource {
public:
    virtual ~Resource() = 0;
};

inline Resource::~Resource() = default;

#pragma once

#include <spdlog/spdlog.h>
#include <memory>

class Logger {
public:
    static Logger& GetInstance() {
        static Logger instance;
        return instance;
    }

    Logger(Logger const&) = delete;
    Logger& operator=(Logger const&) = delete;

    inline std::shared_ptr<spdlog::logger>& GetLogger() { return m_logger; }

private:
    Logger();
    ~Logger() = default;

    std::shared_ptr<spdlog::logger> m_logger;
};

#define LOG_INFO(...) Logger::GetInstance().GetLogger()->info(__VA_ARGS__)
#define LOG_WARN(...) Logger::GetInstance().GetLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...) Logger::GetInstance().GetLogger()->error(__VA_ARGS__)
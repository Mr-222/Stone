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

    const std::shared_ptr<spdlog::logger>& GetLogger() const { return m_logger; }

private:
    Logger();
    ~Logger() = default;

    std::shared_ptr<spdlog::logger> m_logger;
};

#ifndef NDEBUG

#define LOG_INFO(fmt, ...) Logger::GetInstance().GetLogger()->info(fmt __VA_OPT__(,) __VA_ARGS__)
#define LOG_WARN(fmt, ...) Logger::GetInstance().GetLogger()->warn(fmt __VA_OPT__(,) __VA_ARGS__)
#define LOG_ERROR(fmt, ...) Logger::GetInstance().GetLogger()->error(fmt __VA_OPT__(,) __VA_ARGS__)

#define LOG_INFO_IF(condition, fmt, ...) \
    do { \
        if (condition) { \
            Logger::GetInstance().GetLogger()->info(fmt __VA_OPT__(,) __VA_ARGS__); \
        } \
    } while (0)

#define LOG_WARN_IF(condition, fmt, ...) \
    do { \
        if (condition) { \
            Logger::GetInstance().GetLogger()->warn(fmt __VA_OPT__(,) __VA_ARGS__); \
        } \
    } while (0)

#define LOG_ERROR_IF(condition, fmt, ...) \
    do { \
        if (condition) { \
            Logger::GetInstance().GetLogger()->error(fmt __VA_OPT__(,) __VA_ARGS__); \
        } \
    } while (0)

#else

#define LOG_INFO(...)                do {} while (0)
#define LOG_WARN(...)                do {} while (0)
#define LOG_ERROR(...)               do {} while (0)

#define LOG_INFO_IF(condition, ...)  do {} while (0)
#define LOG_WARN_IF(condition, ...)  do {} while (0)
#define LOG_ERROR_IF(condition, ...) do {} while (0)

#endif
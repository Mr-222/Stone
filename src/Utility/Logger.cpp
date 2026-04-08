#include "Utility/Logger.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <vector>

Logger::Logger() {
    auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    // %^ = start color, %T = time, %n = logger name, %l = log level, %v = actual message, %$ = end color
    consoleSink->set_pattern("%^[%T] [%n] [%l]: %v%$");

    auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("Stone.log", true);
    fileSink->set_pattern("[%T] [%l] %n: %v");

    std::vector<spdlog::sink_ptr> sinks { consoleSink, fileSink };

    m_logger = std::make_shared<spdlog::logger>("STONE LOGGER", sinks.begin(), sinks.end());

    m_logger->set_level(spdlog::level::info);
    m_logger->flush_on(spdlog::level::err);

    spdlog::register_logger(m_logger);
}

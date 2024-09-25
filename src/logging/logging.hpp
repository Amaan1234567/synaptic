#ifndef SYNAPTIC_LOGGING_HPP
#define SYNAPTIC_LOGGING_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <mutex>
#include <filesystem>



#ifndef SYNAPTIC_LOG_FILE
#define SYNAPTIC_LOG_FILE std::string_view("~/Documents/synaptic/logs.log")
#endif

#ifndef SYNAPTIC_LOG_LEVEL
#define SYNAPTIC_LOG_LEVEL 0  // Default to WARNING level
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, CRITICAL = 4 };

class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void log(LogLevel level, std::string_view className, std::string_view message) {
        if (static_cast<int>(level) >= SYNAPTIC_LOG_LEVEL) {
            initLog();
            std::lock_guard<std::mutex> lock(m_mutex);
            std::string output = "[" + getTimestamp() + "][PID:" + getProcessID() +
                                 "][TID:" + getThreadID() + "][synaptic log][" + std::string(className) +
                                 "][" + logLevelToString(level) + "]: " + std::string(message);
            
            std::cout << output << std::endl;
            /* while(!m_logFileStream.is_open()) {
                initLog();
            } */
            if (m_logFileStream.is_open()) {
                m_logFileStream << output << std::endl;
            }
            else
            std::cout << "file not open" << std::endl;

            
        }
    }

    // These methods are now public and can be called from anywhere
    void initLog() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::cout << sizeof(SYNAPTIC_LOG_FILE) << std::endl;
        if (sizeof(SYNAPTIC_LOG_FILE) && !m_logFileStream.is_open()) {
            fs::path logPath(SYNAPTIC_LOG_FILE);
            m_logFileStream.open(logPath, std::ios::app);
            if (!m_logFileStream.is_open()) {
                std::cerr << "Error opening log file: " << logPath << std::endl;
            }
        }
    }

    void closeLog() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_logFileStream.is_open()) {
            m_logFileStream.close();
        }
    }

private:
    Logger() { initLog(); }
    ~Logger() { closeLog(); }
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::ofstream m_logFileStream;
    std::mutex m_mutex;

    static const char* logLevelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }

    static std::string getTimestamp() {
        std::time_t now = std::time(nullptr);
        char buf[20];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buf);
    }

    static std::string getThreadID() {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        return oss.str();
    }

    static std::string getProcessID() {
        #ifdef _WIN32
            return std::to_string(GetCurrentProcessId());
        #else
            return std::to_string(getpid());
        #endif
    }
};

// Convenience macros for logging
#define LOG_DEBUG(className, message) Logger::getInstance().log(LogLevel::DEBUG, className, message)
#define LOG_INFO(className, message) Logger::getInstance().log(LogLevel::INFO, className, message)
#define LOG_WARNING(className, message) Logger::getInstance().log(LogLevel::WARNING, className, message)
#define LOG_ERROR(className, message) Logger::getInstance().log(LogLevel::ERROR, className, message)
#define LOG_CRITICAL(className, message) Logger::getInstance().log(LogLevel::CRITICAL, className, message)

#endif // SYNAPTIC_LOGGING_HPP
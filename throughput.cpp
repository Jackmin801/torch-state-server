#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>

void optimize_socket(int sock) {
    int send_buffer_size = 4 * 1024 * 1024;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &send_buffer_size, sizeof(send_buffer_size)) < 0) {
        std::cerr << "Failed to set send buffer: " << strerror(errno) << std::endl;
    }

    int recv_buffer_size = 4 * 1024 * 1024;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &recv_buffer_size, sizeof(recv_buffer_size)) < 0) {
        std::cerr << "Failed to set receive buffer: " << strerror(errno) << std::endl;
    }

    int flag = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) < 0) {
        std::cerr << "Failed to set TCP_NODELAY: " << strerror(errno) << std::endl;
    }
}

void server(size_t N) {
    std::cerr << "Server starting with N = " << N << std::endl;
    
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Server socket creation failed: " << strerror(errno) << std::endl;
        return;
    }

    optimize_socket(server_fd);

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        std::cerr << "Server setsockopt failed: " << strerror(errno) << std::endl;
        close(server_fd);
        return;
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(12348);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Server bind failed: " << strerror(errno) << std::endl;
        close(server_fd);
        return;
    }

    if (listen(server_fd, 1) < 0) {
        std::cerr << "Server listen failed: " << strerror(errno) << std::endl;
        close(server_fd);
        return;
    }

    std::cerr << "Server waiting for connection..." << std::endl;

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_socket = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_socket < 0) {
        std::cerr << "Server accept failed: " << strerror(errno) << std::endl;
        close(server_fd);
        return;
    }

    std::cerr << "Server accepted connection" << std::endl;
    optimize_socket(client_socket);

    // Ensure minimum chunk size is 1
    const size_t chunk_size = std::max(size_t(1), std::min(N, size_t(4 * 1024 * 1024)));
    std::vector<char> payload(chunk_size, '0');
    size_t total_sent = 0;

    while (total_sent < N) {
        size_t remaining = N - total_sent;
        size_t current_chunk = std::min(remaining, chunk_size);
        
        ssize_t sent = send(client_socket, payload.data(), current_chunk, 0);
        if (sent < 0) {
            std::cerr << "Server send failed: " << strerror(errno) << std::endl;
            break;
        }
        total_sent += sent;
        
        // Only show progress for larger transfers
        if (N > 1000000 && total_sent % (N/10) == 0) {
            std::cerr << "Server progress: " << (total_sent * 100.0 / N) << "%" << std::endl;
        }
    }

    std::cerr << "Server finished sending " << total_sent << " bytes" << std::endl;
    close(client_socket);
    close(server_fd);
}

void client(size_t N) {
    std::cerr << "Client starting with N = " << N << std::endl;
    
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Client socket creation failed: " << strerror(errno) << std::endl;
        return;
    }

    optimize_socket(sock);

    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(12348);
    
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Client address conversion failed: " << strerror(errno) << std::endl;
        close(sock);
        return;
    }

    std::cerr << "Client attempting connection..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Client connection failed: " << strerror(errno) << std::endl;
        close(sock);
        return;
    }

    std::cerr << "Client connected" << std::endl;

    // Ensure minimum buffer size is 1
    const size_t buffer_size = std::max(size_t(1), std::min(N, size_t(4 * 1024 * 1024)));
    std::vector<char> buffer(buffer_size);
    size_t total_received = 0;

    while (total_received < N) {
        size_t remaining = N - total_received;
        size_t current_chunk = std::min(remaining, buffer_size);
        
        ssize_t bytes_received = recv(sock, buffer.data(), current_chunk, 0);
        if (bytes_received <= 0) {
            if (bytes_received == 0) {
                std::cerr << "Client connection closed by server" << std::endl;
            } else {
                std::cerr << "Client receive failed: " << strerror(errno) << std::endl;
            }
            close(sock);
            return;
        }

        total_received += bytes_received;
        
        // Only show progress for larger transfers
        if (N > 1000000 && total_received % (N/10) == 0) {
            std::cerr << "Client progress: " << (total_received * 100.0 / N) << "%" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();
    
    double mb_per_sec = (N / (1024.0 * 1024.0)) / duration;
    std::cout << N << " bytes: " << mb_per_sec << " MB/s" << std::endl;
    std::cerr << "Client finished receiving " << total_received << " bytes" << std::endl;

    close(sock);
}

int main() {
    std::cerr << "Starting throughput test..." << std::endl;
    
    for (int i = 0; i < 9; i++) {
        size_t N = static_cast<size_t>(std::pow(10, i));
        std::cerr << "\nTesting with N = " << N << std::endl;
        
        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "Fork failed: " << strerror(errno) << std::endl;
            return 1;
        }
        
        if (pid == 0) {  // Child process
            server(N);
            return 0;
        } else {  // Parent process
            sleep(1);  // Wait for server to start
            client(N);
            int status;
            waitpid(pid, &status, 0);
            if (WIFEXITED(status)) {
                std::cerr << "Server exited with status " << WEXITSTATUS(status) << std::endl;
            } else {
                std::cerr << "Server terminated abnormally" << std::endl;
            }
        }
    }
    return 0;
}
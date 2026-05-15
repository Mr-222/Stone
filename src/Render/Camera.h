#pragma once

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

enum class Movement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

struct CameraConfig {
    glm::vec3 position;
    float yaw;
    float pitch;
    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;
};

// Use left hand basis
class Camera {
public:
    Camera();
    explicit Camera(CameraConfig config);

    glm::mat4 GetViewMatrix() const;
    glm::mat4 GetProjectionMatrix() const;
    void ProcessKeyboard(Movement direction, float deltaTime);
    void ProcessMouseMovement(float xOffset, float yOffset);

private:
    void SetRotationFromYawPitch(float yaw, float pitch);
    void UpdateCameraVectors();

    glm::vec3 m_position;
    glm::vec3 m_front;
    glm::vec3 m_up;
    glm::vec3 m_right;
    glm::quat m_rotation;
    float m_movementSpeed = 3.f;
    float m_mouseSensitivity = .1f;

    float m_fov;
    float m_aspectRatio; // width / height
    float m_nearPlane;
    float m_farPlane;
};

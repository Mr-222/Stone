#include "Camera.h"

constexpr glm::vec3 kLocalRight(1.0f, 0.0f, 0.0f);
constexpr glm::vec3 kLocalUp(0.0f, 1.0f, 0.0f);
constexpr glm::vec3 kLocalForward(0.0f, 0.0f, 1.0f);
constexpr glm::vec3 kWorldUp(0.0f, 1.0f, 0.0f);

Camera::Camera() :
    m_position(glm::vec3(0.0f, 0.0f, 0.0f)),
    m_rotation(glm::identity<glm::quat>()),
    m_fov(45.0f),
    m_aspectRatio(16.0f / 9.0f),
    m_nearPlane(0.01f),
    m_farPlane(100.0f)
{
    UpdateCameraVectors();
}

Camera::Camera(CameraConfig config) {
    m_position = config.position;
    SetRotationFromYawPitch(config.yaw, config.pitch);
    m_fov = config.fov;
    m_aspectRatio = config.aspectRatio;
    m_nearPlane = config.nearPlane;
    m_farPlane = config.farPlane;
    UpdateCameraVectors();
}

glm::mat4 Camera::GetViewMatrix() const {
    return glm::lookAtLH(m_position, m_position + m_front, m_up);
}

glm::mat4 Camera::GetProjectionMatrix() const {
    return glm::perspectiveLH_ZO(glm::radians(m_fov), m_aspectRatio, m_nearPlane, m_farPlane);
}

void Camera::ProcessKeyboard(Movement direction, float deltaTime) {
    float velocity = m_movementSpeed * deltaTime;
    if (direction == Movement::FORWARD)
        m_position += m_front * velocity;
    if (direction == Movement::BACKWARD)
        m_position -= m_front * velocity;
    if (direction == Movement::LEFT)
        m_position -= m_right * velocity;
    if (direction == Movement::RIGHT)
        m_position += m_right * velocity;
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset) {
    xoffset *= m_mouseSensitivity;
    yoffset *= m_mouseSensitivity;

    const glm::quat yaw = glm::angleAxis(glm::radians(xoffset), kWorldUp);
    const glm::quat yawedRotation = glm::normalize(yaw * m_rotation);
    const glm::vec3 pitchAxis = glm::normalize(yawedRotation * kLocalRight);
    const glm::quat pitch = glm::angleAxis(glm::radians(yoffset), pitchAxis);
    m_rotation = glm::normalize(pitch * yawedRotation);

    UpdateCameraVectors();
}

void Camera::SetRotationFromYawPitch(float yaw, float pitch) {
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    m_rotation = glm::quatLookAtLH(glm::normalize(front), kWorldUp);
}

void Camera::UpdateCameraVectors() {
    m_front = glm::normalize(m_rotation * kLocalForward);
    m_right = glm::normalize(m_rotation * kLocalRight);
    m_up = glm::normalize(m_rotation * kLocalUp);
}

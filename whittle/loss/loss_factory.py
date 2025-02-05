import torch
import torch.nn.functional as F
from whittle.loss.kd_losses import (
    forward_kl,
    reverse_kl,
    symmetric_kl,
    js_distance,
    l1_loss,
    l2_loss,
    cosine_similarity,
    mmd_loss,
)


class LossFactory:
    def __init__(self, alpha=0.5, beta=0.5, temperature=1.0, weight_scheme="other"):
        """
        Initializes the LossFactory with a given alpha and temperature.

        Args:
            alpha: Weight for the cross-entropy loss with ground truth labels.
            temperature: Temperature for distillation.
        """
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta
        self.weight_scheme = weight_scheme

    def cross_entropy_loss(self, logits, labels):
        """
        Computes the standard cross-entropy loss with ground truth labels.

        Args:
            logits: Student model logits.
            labels: Ground truth labels.

        Returns:
            Cross-entropy loss.
        """
        return F.cross_entropy(logits, labels)

    def distillation_loss(self, logits, teacher_logits, loss_type="forward_kl"):
        """
        Computes the distillation loss between student and teacher logits.

        Args:
            logits: Student model logits.
            teacher_logits: Teacher model logits.
            loss_type: Type of distillation loss. Supported values:
                       - "forward_kl"
                       - "reverse_kl"
                       - "symmetric_kl"
                       - "js_distance"
                       - "l1"
                       - "l2"
                       - "cosine_similarity"
                       - "mmd"

        Returns:
            Distillation loss between student and teacher.
        """
        if loss_type == "forward_kl":
            return forward_kl(logits, teacher_logits, self.temperature)
        elif loss_type == "reverse_kl":
            return reverse_kl(logits, teacher_logits, self.temperature)
        elif loss_type == "symmetric_kl":
            return symmetric_kl(logits, teacher_logits, self.temperature)
        elif loss_type == "js_distance":
            return js_distance(logits, teacher_logits, self.temperature)
        elif loss_type == "l1":
            return l1_loss(logits, teacher_logits)
        elif loss_type == "l2":
            return l2_loss(logits, teacher_logits)
        elif loss_type == "cosine_similarity":
            return cosine_similarity(logits, teacher_logits, self.temperature)
        elif loss_type == "mmd":
            return mmd_loss(logits, teacher_logits)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def compute_loss(self, logits, teacher_logits, labels, loss_type="forward_kl"):
        """
        Combines cross-entropy loss with distillation loss.

        Args:
            logits: Student model logits.
            teacher_logits: Teacher model logits.
            labels: Ground truth labels.
            loss_type: Type of distillation loss to combine with cross-entropy.

        Returns:
            Combined loss: alpha * cross_entropy + (1 - alpha) * distillation_loss
        """
        # Cross-entropy with ground truth
        ce_loss = self.cross_entropy_loss(logits, labels)

        # Distillation loss
        distil_loss = self.distillation_loss(logits, teacher_logits, loss_type)

        # Combined loss: alpha * CE + (1 - alpha) *
        if self.weight_scheme == "default":
            coefficient1 = 1
            if distil_loss == 0:
                coefficient2 = 1
            else:
                coefficient2 = ce_loss.detach().item() / distil_loss.detach().item()
        else:
            coefficient1 = self.alpha
            coefficient2 = self.beta
        combined_loss = coefficient1 * ce_loss + coefficient2 * distil_loss
        return combined_loss


# Example usage of the LossFactory
if __name__ == "__main__":
    from litgpt import LLM

    # Assume logits, teacher_logits, and labels are provided
    llm_student = LLM.load("EleutherAI/pythia-1b").cuda()
    llm_teacher = LLM.load("EleutherAI/pythia-2.8b").cuda()
    # Define input tensor
    input_tensor = torch.randint(
        0, 50304, (32, 128)
    ).cuda()  # Input shape: (batch_size, sequence_length)

    # Get outputs from the models
    output_student = llm_student(input_tensor).cuda()  # Output shape: (32, 128, 50304)
    output_teacher = llm_teacher(input_tensor).cuda()  # Output shape: (32, 128, 50304)

    # Labels should be of shape (32, 128)
    labels = torch.randint(0, 50304, (32, 128)).cuda()  # Ground truth labels

    # Initialize the loss factory with alpha=0.7 and temperature=1.0
    loss_factory = LossFactory(alpha=0.7, temperature=2.0)

    # Compute the combined loss using forward KL divergence as the distillation loss
    for loss in [
        "forward_kl",
        "reverse_kl",
        "symmetric_kl",
        "js_distance",
        "l1",
        "l2",
        "cosine_similarity",
        "mmd",
    ]:
        with torch.no_grad():
            combined_loss = loss_factory.compute_loss(
                output_student.view(-1, 50304),
                output_teacher.view(-1, 50304),
                labels.view(-1),
                loss_type=loss,
            )
            print("Loss type: ", loss)
            print("Combined Loss:", combined_loss.item())
    # Assume logits, teacher_logits, and labels are
    torch.cuda.empty_cache()
    llm_student = LLM.load("EleutherAI/pythia-410m").cuda()
    llm_teacher = LLM.load("EleutherAI/pythia-1b").cuda()

    # Get outputs from the models
    output_student = llm_student(input_tensor).cuda()  # Output shape: (32, 128, 50304)
    output_teacher = llm_teacher(input_tensor).cuda()  # Output shape: (32, 128, 50304)

    # Labels should be of shape (32, 128)

    # Initialize the loss factory with alpha=0.7 and temperature=1.0
    loss_factory = LossFactory(alpha=0.7, temperature=2.0)

    # Compute the combined loss using forward KL divergence as the distillation loss
    for loss in [
        "forward_kl",
        "reverse_kl",
        "symmetric_kl",
        "js_distance",
        "l1",
        "l2",
        "cosine_similarity",
        "mmd",
    ]:
        with torch.no_grad():
            combined_loss = loss_factory.compute_loss(
                output_student.view(-1, 50304),
                output_teacher.view(-1, 50304),
                labels.view(-1),
                loss_type=loss,
            )
        print("Loss type: ", loss)
        print("Combined Loss:", combined_loss.item())

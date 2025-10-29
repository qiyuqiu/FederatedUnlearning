import torch


def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy


def predict_with_shards(input_data, shard_connections, device):
    """
    Predict the result of input data using multiple shard models and their associated weights.

    :param input_data: Input data, of type torch.Tensor
    :param shard_connections: A list of shards, each containing a model and its weight
    :param device: Device to run the computation on, e.g., 'cuda' or 'cpu'
    :return: Final predicted result
    """
    input_data = input_data.to(device)
    total_weight = sum(shard_connection.weight for shard_connection in shard_connections)  # Total weight for normalization
    votes = None  # Initialize vote tensor

    # Iterate over all models, gather and weight their predictions
    for shard_connection in shard_connections:
        model = shard_connection.model
        weight = shard_connection.weight
        model.to(device)  # Move model to the specified device
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            outputs, _ = model(input_data)  # Get model output
            _, predicted_class = torch.max(outputs, dim=1)  # Get predicted class
            weighted_votes = torch.zeros(outputs.shape).to(device)  # Initialize weighted vote tensor
            for i in range(outputs.shape[0]):
                weighted_votes[i, predicted_class[i]] = weight / total_weight  # Apply weighted vote

            if votes is None:
                votes = weighted_votes
            else:
                votes += weighted_votes

    # Compute final prediction from weighted votes
    _, final_predicted_class = torch.max(votes, dim=1)
    return final_predicted_class.cpu().numpy()  # Return the predicted class as a NumPy array

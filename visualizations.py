coordinates = np.array(dataset.y_data)
uniq_coordinates = np.unique(coordinates, axis=0)

plt.scatter(uniq_coordinates[:, 0].tolist(), uniq_coordinates[:, 1].tolist())
plt.savefig("visualizations/temp.png")